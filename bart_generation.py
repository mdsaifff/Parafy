import argparse
import ast
import itertools
import json
import math
import os
import random
import re
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from sacrebleu.metrics import BLEU



MODEL_NAME = "facebook/bart-large"

ROOT = os.path.dirname(os.path.abspath(__file__))   # so the scripts work from any working directory

SEGMENT_COLUMN = "sentence1_segment_location"

SOURCE_MAX_LENGTH = 64   # just the source sentence, tokenized on its own for the copy penalty

RUNS_DIR = os.path.join(ROOT, "bart_generation_runs")   # one folder per run: the checkpoint and metrics.json. predictions/ keeps only the test predictions

SUBMISSION_PATH = os.path.join(ROOT, "predictions", "etpc-paraphrase-generation-test-output.csv")

RERANK_WEIGHTS = {"model": 0.45, "source": 0.20, "type": 0.20, "fluency": 0.15, "temperature": 0.05}

# type id to type name, taken from the jpwahle/etpc dataset
with open(os.path.join(ROOT, "data", "etpc_type_names.json"), encoding="utf-8") as file:
    ETPC_TYPE_NAMES = {int(key): value for key, value in json.load(file).items()}


def parse_type_ids(value):
    """read the list of type ids stored in the csv"""
    return [int(item) for item in ast.literal_eval(value)]


def normalize_sentence(text):
    return " ".join(str(text).lower().split())


def remove_leaky_training_rows(train_df, dev_df):
    """drop train rows that also show up in dev, reversed pairs included"""
    dev_ids = set(dev_df["id"])
    dev_sentences = set(dev_df["sentence1"].map(normalize_sentence)) | set(dev_df["sentence2"].map(normalize_sentence))
    leaky = ( train_df["id"].isin(dev_ids) | train_df["sentence1"].map(normalize_sentence).isin(dev_sentences) | train_df["sentence2"].map(normalize_sentence).isin(dev_sentences) )
    clean_train = train_df.loc[~leaky].reset_index(drop=True)   # renumber the rows we kept
    print(f"Removed {len(train_df) - len(clean_train)} train examples overlapping with dev.")
    return clean_train


def reverse_augment(dataset):
    """add swapped sentence pairs to the training data

    id 25 is one combined "addition/deletion" label so it still fits after a
    swap! entailment (id 31) only holds in one direction, so those rows are
    left out of the swapped copy :))
    """
    reversible = dataset["paraphrase_type_ids"].apply(lambda value: 31 not in parse_type_ids(value))
    reversed_df = dataset.loc[reversible].copy()
    reversed_df["sentence1"], reversed_df["sentence2"] = ( reversed_df["sentence2"].values, reversed_df["sentence1"].values )
    reversed_df[SEGMENT_COLUMN] = reversed_df["sentence2_segment_location"].values   # the spans now come from the other sentence
    reversed_df["id"] = reversed_df["id"] + "-reverse"
    skipped = len(dataset) - len(reversed_df)
    print(f"Reverse augmentation: added {len(reversed_df)} reversed pairs ({skipped} Entailment-tagged rows skipped).")
    return pd.concat([dataset, reversed_df], ignore_index=True)


def type_instruction(type_ids):
    if len(type_ids) > 1:
        type_ids = [type_id for type_id in type_ids if type_id != 29]  # id 29 means nothing changed, so prefer any other type
    names = [ETPC_TYPE_NAMES[type_id] for type_id in type_ids]
    return "; ".join(dict.fromkeys(names)) or "preserve the meaning with different wording"


def segment_text(value):
    """read the segment spans, which the csv stores as a list inside a string"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        return " ".join(str(item) for item in ast.literal_eval(str(value)))
    except (ValueError, SyntaxError):
        return str(value)


def make_input(row, prompt="natural", use_segments=True, sep_token="</s>"):
    """build the input prompt, using the template from wahle et al

    the type ids become real words that bart has seen before, instead of the
    plain format sentence1 + sep + spans + sep + raw ids. --prompt sep builds
    that plain format, so the ablation can compare the two.
    """
    segments = segment_text(getattr(row, SEGMENT_COLUMN, None)) if use_segments else ""
    if prompt == "sep":
        raw_ids = " ".join(str(type_id) for type_id in parse_type_ids(row.paraphrase_type_ids))
        return f"{row.sentence1} {sep_token} {segments} {sep_token} {raw_ids}"
    types = ", ".join(type_instruction(parse_type_ids(row.paraphrase_type_ids)).split("; "))
    segment_line = f"Segment Location: {segments}\n" if segments else ""
    return ( "Given the following sentence, generate a paraphrase with the following types.\n"
             f"Sentence: {row.sentence1}\n"
             f"{segment_line}"
             f"Paraphrase Types: {types}" )


def transform_data(dataset, tokenizer, max_length=256, batch_size=16, shuffle=False,
                   prompt="natural", use_segments=True):
    input_texts = [make_input(row, prompt, use_segments, tokenizer.sep_token) for row in dataset.itertuples()]
    encodings = tokenizer(input_texts, max_length=max_length, truncation=True, padding="longest", return_tensors="pt")
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]   # mask is 1 for a real token, 0 for padding
    # keep the plain source apart, the copy penalty must not see the prompt
    # text, the span numbers or the type names
    source_ids = tokenizer( dataset["sentence1"].astype(str).tolist(), max_length=SOURCE_MAX_LENGTH, truncation=True,
                            padding="longest", add_special_tokens=False, return_tensors="pt")["input_ids"]
    if "sentence2" in dataset.columns:
        labels = tokenizer( dataset["sentence2"].tolist(), max_length=max_length, truncation=True,
                            padding="longest", return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100   # ignore padding when we compute the loss
    else:
        labels = torch.zeros_like(input_ids)
    index = torch.arange(input_ids.size(0))   # lets a shuffled batch find its cached brio candidates
    return DataLoader(TensorDataset(input_ids, attention_mask, labels, source_ids, index),
                      batch_size=batch_size, shuffle=shuffle)


COPY_PENALTY_STOPWORDS = { "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but", "is", "are",
                           "was", "were", "be", "been", "by", "with", "as", "that", "it", "its", "this", "these",
                           "from", "has", "have", "had", "not", "s", "t" }


def build_canonical_map(tokenizer):
    """map "The", "the" and " the" to one id, and mark tokens the penalty skips

    the source sits in the middle of the prompt, so its first word gets a
    different id than the same word right after bos in the label. without this
    the penalty fired on every word
    """
    vocab = tokenizer.get_vocab()
    size = max(vocab.values()) + 1
    canonical = torch.arange(size, dtype=torch.long)
    ignore = torch.zeros(size, dtype=torch.bool)
    specials = set(tokenizer.all_special_tokens)
    for piece, token_id in vocab.items():
        word = piece.lstrip("Ġ").lower()          # Ġ marks a leading space in bpe
        target = vocab.get("Ġ" + word, vocab.get(word))
        if target is not None:
            canonical[token_id] = target
        # a paraphrase may keep "the", so stopwords and punctuation are skipped
        if piece in specials or not any(char.isalnum() for char in word) or word in COPY_PENALTY_STOPWORDS:
            ignore[token_id] = True
    return canonical, ignore


def label_smoothed_loss(logits, labels, smoothing=0.1):
    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)   # put 0 where the label is ignored
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)   # negative log probability
    smooth = -log_probs.mean(dim=-1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).masked_select(valid).mean()


def copy_penalty_loss(logits, labels, source_ids, pad_token_id, canonical, ignore_mask):
    """push down source words that the reference did not keep

    both sides are normalized first, so we only punish words that were really
    dropped, not words that only look different after bpe
    """
    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, pad_token_id)
    in_target = (canonical[source_ids].unsqueeze(2) == canonical[safe_labels].unsqueeze(1)).any(dim=2)   # the source word also shows up in the target
    # a word that repeats in the source used to be counted once per copy, which
    # pushed copy_prob above 1 and left the loss flat with no gradient. keeping
    # only the first spot of each id makes the sum run over distinct ids
    position = torch.arange(source_ids.size(1), device=source_ids.device)
    seen_earlier = ((source_ids.unsqueeze(2) == source_ids.unsqueeze(1))
                    & (position.unsqueeze(0) < position.unsqueeze(1))).any(dim=2)
    penalize = (~in_target) & source_ids.ne(pad_token_id) & ~ignore_mask[source_ids] & ~seen_earlier   # only the content words a paraphrase should have dropped
    log_probs = F.log_softmax(logits, dim=-1)
    source_positions = source_ids.unsqueeze(1).expand(-1, logits.size(1), -1)   # look up the source ids at every decoder step
    copy_prob = (log_probs.gather(-1, source_positions).exp() * penalize.unsqueeze(1)).sum(dim=-1)
    return -torch.log1p(-copy_prob.clamp(max=1.0 - 1e-6)).masked_select(valid).mean()   # clamping copy_prob keeps a gradient below the cap


def sequence_log_likelihood(logits, labels):
    """average log likelihood of each whole sentence"""
    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_scores = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_scores * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)   # clamp_min so we never divide by zero


REPEATED_WORD = re.compile(r"\b(\w+)( \1\b){2,}", re.IGNORECASE)


def is_degenerate(text):
    """spot text that games the metric, such as a sentence with its start cut off

    a short fragment scores well on reference_bleu * (100 - input_bleu) so the
    brio reward and the reranker both throw such text away!! without this the
    model learns to game the score instead of paraphrasing
    """
    text = str(text)
    return ( not text.strip()
             or text[:1].isspace()   # bart only starts with a space when the target began mid sentence
             or next((char for char in text.strip() if char.isalpha()), "").islower()
             or bool(REPEATED_WORD.search(text)) )   # "in a statement in part in in in"


def sentence_bleu(candidate, reference):
    return BLEU(effective_order=True).sentence_score(candidate, [reference]).score / 100.0


def candidate_quality(candidate, reference, source):
    """close to the reference and far from the input, the shape of penalized bleu"""
    if is_degenerate(candidate):
        return 0.0
    return sentence_bleu(candidate, reference) * (1.0 - sentence_bleu(candidate, source))


def generate_brio_candidates(model, dataloader, device, args):
    """build the brio candidate pool once per epoch, the way the paper does

    decoding matches inference, so the loss ranks the candidates the model
    will really produce
    """
    model.eval()
    width = args.max_new_tokens + 2
    pools = []
    with torch.no_grad():
        for input_ids, attention_mask, _, _, _ in tqdm(dataloader, desc="brio-candidates"):
            generated = model.generate( input_ids.to(device), attention_mask=attention_mask.to(device),
                                        num_beams=max(args.num_candidates, 2), num_return_sequences=args.num_candidates,
                                        max_new_tokens=args.max_new_tokens, length_penalty=args.length_penalty,
                                        no_repeat_ngram_size=args.no_repeat_ngram_size, early_stopping=True )
            keep = min(width, generated.size(1))
            padded = torch.full((generated.size(0), width), model.config.pad_token_id, dtype=torch.long)
            padded[:, :keep] = generated[:, :keep].detach().cpu()
            pools.append(padded.view(input_ids.size(0), args.num_candidates, width))
    model.train()
    return torch.cat(pools, dim=0)


def brio_ranking_loss(model, input_ids, attention_mask, labels, source_ids, candidate_ids, tokenizer, num_candidates):
    """rank each pair of cached candidates by penalized bleu quality, brio style

    a margin loss teaches bart to score the better candidate higher and the margin grows with the quality gap
    """
    flat_candidates = candidate_ids.reshape(-1, candidate_ids.size(-1)).to(input_ids.device)
    candidate_labels = flat_candidates[:, 1:].clone()   # the first token is bart's decoder start token, not a real target
    candidate_labels[candidate_labels == tokenizer.pad_token_id] = -100
    repeated_inputs = input_ids.repeat_interleave(num_candidates, dim=0)   # repeat each input n times
    repeated_masks = attention_mask.repeat_interleave(num_candidates, dim=0)

    model.eval()   # dropout off, otherwise the ranking signal is mostly sampling noise
    candidate_outputs = model(input_ids=repeated_inputs, attention_mask=repeated_masks, labels=candidate_labels)
    model.train()
    candidate_scores = sequence_log_likelihood(candidate_outputs.logits, candidate_labels).view(input_ids.size(0), num_candidates)

    reference_ids = labels.masked_fill(labels.eq(-100), tokenizer.pad_token_id).detach().cpu()
    references = tokenizer.batch_decode(reference_ids, skip_special_tokens=True)   # back to plain text
    # plain sentence1 straight from the batch, so this works with any prompt
    sources = tokenizer.batch_decode(source_ids.detach().cpu(), skip_special_tokens=True)
    candidates = tokenizer.batch_decode(flat_candidates.detach().cpu(), skip_special_tokens=True)
    quality = torch.tensor( [candidate_quality(candidate, references[index // num_candidates], sources[index // num_candidates])
                             for index, candidate in enumerate(candidates)],
                            device=input_ids.device).view(input_ids.size(0), num_candidates)

    losses = []
    for scores, qualities in zip(candidate_scores, quality):
        for i, j in itertools.combinations(range(num_candidates), 2):   # every pair of candidates
            quality_gap = qualities[i] - qualities[j]
            if quality_gap == 0:
                continue
            better, worse = (i, j) if quality_gap > 0 else (j, i)
            margin = 0.05 + 0.20 * abs(quality_gap)   # how much better the model should score the better candidate
            losses.append(F.relu(margin - (scores[better] - scores[worse])))   # penalize when the ranking is not good enough
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=input_ids.device)

# reranking

def word_stats(source, candidate):
    source_words = re.findall(r"\w+", normalize_sentence(source))
    candidate_words = re.findall(r"\w+", normalize_sentence(candidate))
    overlap = len(set(source_words) & set(candidate_words)) / max(1, len(set(source_words) | set(candidate_words)))
    exact_copy = normalize_sentence(source) == normalize_sentence(candidate)
    bigrams = list(zip(candidate_words, candidate_words[1:]))
    repetition = 1.0 - len(set(bigrams)) / max(1, len(bigrams))
    length_ratio = len(candidate_words) / max(1, len(source_words))
    return overlap, exact_copy, repetition, length_ratio


def instruction_agreement(source, candidate, instruction, overlap, exact_copy, length_ratio):
    """how well the candidate makes the change the types asked for"""
    instruction = instruction.lower()
    if "identity" in instruction:
        return 1.0 if exact_copy else overlap
    if "addition" in instruction or "deletion" in instruction:
        # best near a 25% length change and fades past 60%, so a cut off fragment does not score 1.0
        return math.exp(-((abs(length_ratio - 1.0) - 0.25) ** 2) / (2 * 0.20 ** 2))
    if any(word in instruction for word in ("order", "structure", "diathesis", "subordination", "coordination")):
        order_change = 1.0 - SequenceMatcher(None, normalize_sentence(source), normalize_sentence(candidate)).ratio()
        return 0.5 * overlap + 0.5 * order_change   # each gets 50% weight
    return max(0.0, 1.0 - overlap)   # for everything else


def select_candidate(source, instruction, candidates, beam_scores):
    
    temperature = max(1e-3, RERANK_WEIGHTS["temperature"])
    scaled = [score / temperature for score in beam_scores]
    exponentials = [math.exp(value - max(scaled)) for value in scaled]   # plain python softmax over the beam scores
    model_scores = [value / sum(exponentials) for value in exponentials]

    stats = [word_stats(source, candidate) for candidate in candidates]
    usable = [not is_degenerate(candidate) for candidate in candidates]
    # an exact copy scores near zero on the (100 - input_bleu) part, so it is
    # never the right pick while another usable candidate is in the pool
    copy_is_avoidable = any(ok and not stat[1] for ok, stat in zip(usable, stats))

    best_index, best_score = 0, -float("inf")   # falls back to the top beam if every candidate is rejected
    for index, candidate in enumerate(candidates):
        overlap, exact_copy, repetition, length_ratio = stats[index]
        if not usable[index] or (exact_copy and copy_is_avoidable):
            continue
        source_score = max(0.0, 1.0 - overlap / 0.6) - (0.75 if exact_copy else 0.0)   # reward moving away from the input, that is what penalized bleu wants
        fluency = max(0.0, 1.0 - repetition - 0.5 * abs(length_ratio - 1.0))
        agreement = instruction_agreement(source, candidate, instruction, overlap, exact_copy, length_ratio)
        score = ( RERANK_WEIGHTS["model"] * model_scores[index] + RERANK_WEIGHTS["source"] * source_score
                  + RERANK_WEIGHTS["type"] * agreement + RERANK_WEIGHTS["fluency"] * fluency )
        if score > best_score:
            best_index, best_score = index, score
    return candidates[best_index]



def generate_sentences(model, dataloader, dataset, device, tokenizer, args, description="generating", rerank=True):
    """make candidates for every row and rerank them

    rerank=False just keeps the top beam. decoding is the same either way, so
    both can be compared on one checkpoint with no extra training.
    """
    rows = dataset.to_dict("records")
    predictions = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for input_ids, attention_mask, _, _, _ in tqdm(dataloader, desc=description):
            generated = model.generate( input_ids.to(device), attention_mask=attention_mask.to(device),
                num_beams=max(args.num_beams, args.num_candidates), num_return_sequences=args.num_candidates,
                max_new_tokens=args.max_new_tokens, length_penalty=args.length_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size, 
                early_stopping=True,return_dict_in_generate=True, output_scores=True )

            decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
            beam_scores = generated.sequences_scores.detach().cpu().tolist()
            for local_index in range(input_ids.size(0)):
                row = rows[offset + local_index]   # the original dataset row
                start = local_index * args.num_candidates
                candidates = decoded[start : start + args.num_candidates]
                if not rerank:
                    predictions.append(candidates[0])
                    continue
                instruction = type_instruction(parse_type_ids(row["paraphrase_type_ids"]))
                predictions.append(select_candidate( row["sentence1"], instruction, candidates,
                                                     beam_scores[start : start + args.num_candidates] ))
            offset += input_ids.size(0)
    return predictions


def evaluate_model(model, dev_data, dev_df, device, tokenizer, args, rerank=True, label="dev"):
    
    predictions = generate_sentences(model, dev_data, dev_df, device, tokenizer, args, "validation", rerank)
    bleu = BLEU()
    reference_bleu = bleu.corpus_score(dev_df["sentence2"].tolist(), [predictions]).score
    input_bleu = 100.0 - bleu.corpus_score(dev_df["sentence1"].tolist(), [predictions]).score
    penalized_bleu = reference_bleu * input_bleu / 52.0
    print(f"{label}: BLEU {reference_bleu:.3f} | negative input BLEU {input_bleu:.3f} | penalized BLEU {penalized_bleu:.3f}")
    return {"corpus_bleu": reference_bleu, "input_bleu": input_bleu, "penalized_bleu": penalized_bleu}


def train_model(model, train_data, dev_data, dev_df, device, tokenizer, args):
    """fine tune bart with label smoothing, the copy penalty and the brio ranking loss"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6, weight_decay=0.0)
    total_steps = max(1, min(args.num_epochs, args.schedule_epochs) * len(train_data))   # early stopping rarely reaches num_epochs, so decay over a realistic length
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    canonical, ignore_mask = build_canonical_map(tokenizer)
    canonical, ignore_mask = canonical.to(device), ignore_mask.to(device)
    ordered_train = DataLoader(train_data.dataset, batch_size=args.batch_size, shuffle=False)   # stable order, so cached candidates line up with the dataset index
    checkpoint_dir = os.path.join(RUNS_DIR, args.run_name)   # one folder per ablation run
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_bleu, epochs_without_improvement, history = -float("inf"), 0, []

    for epoch in range(args.num_epochs):
        # in the paper brio starts from an already trained model, so wait until
        # the plain loss has had a few epochs on its own
        brio_active = args.brio_weight > 0 and epoch >= args.brio_start_epoch   # no point decoding candidates nobody uses
        brio_cache = generate_brio_candidates(model, ordered_train, device, args) if brio_active else None

        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels, source_ids, index in tqdm(train_data, desc=f"train-epoch-{epoch + 1}"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            labels, source_ids = labels.to(device), source_ids.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = label_smoothed_loss(outputs.logits, labels, args.label_smoothing)
            loss = loss + args.copy_weight * copy_penalty_loss(outputs.logits, labels, source_ids,
                                                               tokenizer.pad_token_id, canonical, ignore_mask)
            if brio_active:
                loss = loss + args.brio_weight * brio_ranking_loss(model, input_ids, attention_mask, labels, source_ids,
                                                                   brio_cache[index], tokenizer, args.num_candidates)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.num_epochs} - loss {total_loss / len(train_data):.4f}")
        metrics = evaluate_model(model, dev_data, dev_df, device, tokenizer, args, label=f"epoch {epoch + 1}")
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_data), "brio_active": brio_active, **metrics})
        if metrics["penalized_bleu"] > best_bleu:
            best_bleu, epochs_without_improvement = metrics["penalized_bleu"], 0
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    print(f"Best dev penalized BLEU: {best_bleu:.3f}")
    return BartForConditionalGeneration.from_pretrained(checkpoint_dir).to(device), history


def test_model(model, test_data, test_df, device, tokenizer, args):
    """make the test paraphrases as an id / Generated_sentence2 table"""
    predictions = generate_sentences(model, test_data, test_df, device, tokenizer, args, "test")
    return pd.DataFrame({"id": test_df["id"].tolist(), "Generated_sentence2": predictions})


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finetune_paraphrase_generation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_segments = not args.no_segments
    print(f"Run {args.run_name} on {device}: prompt {args.prompt} | segments {use_segments} | "
          f"reverse {not args.no_reverse} | label smoothing {args.label_smoothing} | "
          f"copy {args.copy_weight} | brio {args.brio_weight} | seed {args.seed}")

    train_df = pd.read_csv(os.path.join(ROOT, "data", "etpc-paraphrase-train-split.csv"))
    dev_df = pd.read_csv(os.path.join(ROOT, "data", "etpc-paraphrase-dev.csv"))
    test_df = pd.read_csv(os.path.join(ROOT, "data", "etpc-paraphrase-generation-test.csv"))
    rows_before = len(train_df)
    train_df = remove_leaky_training_rows(train_df, dev_df)
    leaky_removed = rows_before - len(train_df)
    if not args.no_reverse:
        train_df = reverse_augment(train_df)
    print(f"Loaded {len(train_df)} training samples.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=args.local_files_only)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME, local_files_only=args.local_files_only).to(device)

    train_data = transform_data(train_df, tokenizer, args.max_length, args.batch_size, True, args.prompt, use_segments)
    dev_data = transform_data(dev_df, tokenizer, args.max_length, args.batch_size, False, args.prompt, use_segments)

    model, history = train_model(model, train_data, dev_data, dev_df, device, tokenizer, args)
    print("Training finished.")

    # the reranking ablation is free: one checkpoint, two ways to pick from the same beams
    with_rerank = evaluate_model(model, dev_data, dev_df, device, tokenizer, args, True, "dev + rerank")
    top_beam = evaluate_model(model, dev_data, dev_df, device, tokenizer, args, False, "dev, top beam")
    print(f"Reranking alone changes penalized BLEU by {with_rerank['penalized_bleu'] - top_beam['penalized_bleu']:+.3f}.")

    results = { "run_name": args.run_name,
                "seed": args.seed,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "config": { "prompt": args.prompt, "segments": use_segments, "reverse": not args.no_reverse,
                            "label_smoothing": args.label_smoothing, "copy_weight": args.copy_weight,
                            "brio_weight": args.brio_weight, "brio_start_epoch": args.brio_start_epoch,
                            "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                            "num_epochs": args.num_epochs, "num_beams": args.num_beams,
                            "num_candidates": args.num_candidates, "length_penalty": args.length_penalty,
                            "no_repeat_ngram_size": args.no_repeat_ngram_size, "max_new_tokens": args.max_new_tokens },
                "data": {"train_rows": len(train_df), "leaky_rows_removed": leaky_removed, "dev_rows": len(dev_df)},
                "dev_rerank": with_rerank,
                "dev_top_beam": top_beam,
                "rerank_delta": with_rerank["penalized_bleu"] - top_beam["penalized_bleu"],
                "best_epoch_penalized_bleu": max(epoch["penalized_bleu"] for epoch in history),
                "epochs": history,
                "submission_written": not args.no_submission }

    if not args.no_submission:
        test_data = transform_data(test_df, tokenizer, args.max_length, args.batch_size, False, args.prompt, use_segments)
        test_output = test_model(model, test_data, test_df, device, tokenizer, args)
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        test_output.to_csv(SUBMISSION_PATH, index=False)
        print(f"Wrote {SUBMISSION_PATH}")
    else:   # do not overwrite the test predictions
        print("Skipped the test split (--no_submission).")

    # all we need to compare runs later, saved next to that run's checkpoint
    metrics_path = os.path.join(RUNS_DIR, args.run_name, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Wrote {metrics_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Parafy: type-controlled paraphrasing with BART (ETPC)")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--local_files_only", action="store_true")
    # turn one improvement off per run to see what it is worth
    parser.add_argument("--run_name", default="full")                    # checkpoint goes to bart_generation_runs/<run_name>
    parser.add_argument("--prompt", choices=("natural", "sep"), default="natural")
    parser.add_argument("--no_reverse", action="store_true")
    parser.add_argument("--no_segments", action="store_true")
    parser.add_argument("--no_submission", action="store_true")          # ablation runs leave the test predictions alone, only the best model writes it
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--schedule_epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--copy_weight", type=float, default=0.5)
    parser.add_argument("--brio_weight", type=float, default=0.2)
    parser.add_argument("--brio_start_epoch", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
