# Paraphrase Type Generation with BART

[![tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

**Controllable paraphrasing:** given a sentence and a list of linguistic *paraphrase types* (e.g. *synthetic/analytic substitution*, *diathesis alternation*, *addition/deletion*), generate a paraphrase that applies those changes.

Fine-tuned `facebook/bart-large` on the ETPC corpus. I combined reverse augmentation, natural-language type prompts, a copy penalty, a **BRIO ranking loss** and type-aware candidate reranking. Together they raised the graded metric (Penalized BLEU) on the dev split **from 18.98 to 34.39 (+81%)**.

![Ablation](assets/ablation.png)

## The problem: copying is the easy way out

A model that copies its input word for word scores ~48 BLEU against the human reference while doing no paraphrasing at all. Plain fine-tuning learns exactly that habit: the baseline's outputs score **78 BLEU against their own input**. The task is therefore graded with **Penalized BLEU**, which rewards closeness to the reference *and* distance from the input:

```
penalized_bleu = BLEU(output, reference) × (100 − BLEU(output, input)) / 52
```

Every technique below is aimed at that one failure mode.

## Method

| Component | What it does |
|---|---|
| **Leak removal** | Drops the 56 train rows whose id or sentence pair (normalised, either direction) also appears in dev |
| **Reverse augmentation** | Adds every pair with source and target swapped (2,128 → 4,210 pairs). The same sentence is now both a source and a target, so copying stops paying off. *Entailment* pairs are skipped because they only hold in one direction |
| **Natural-language type prompts** | Type ids like `11` become words BART has seen (`Synthetic/analytic substitution`), following Wahle et al. (2023) |
| **Label smoothing (0.1)** | There is more than one valid paraphrase, so the model should not be forced to be 100% certain about one |
| **Copy penalty** | An auxiliary loss pushing down the probability of source *content* words that the reference dropped. Tokens are canonicalised (`The`/`the`/`Ġthe` map to one id), and stopwords and repeated ids are ignored; without that the penalty fired on every word |
| **BRIO ranking loss** (Liu et al., 2022) | Each epoch the model decodes 4 candidates per training example. A margin loss teaches it to score candidates with higher (reference BLEU × distance from input) above worse ones. It switches on at epoch 3 |
| **Degenerate-output filter** | Rejects empty outputs, mid-sentence fragments and `in in in` repetition, in both the BRIO reward and the reranker |
| **Type-aware reranking** | Picks the best of 4 beams by model score (45%), distance from source (20%), agreement with the requested type (20%) and fluency (15%) |

All of it lives in [`bart_generation.py`](bart_generation.py).

## Results (dev split, 546 pairs)

Forward, add-one ablation: each row is the `natural_language` baseline with **one** change added. `full` has all of them.

| Run | Penalized BLEU | Δ vs baseline | BLEU vs reference | BLEU vs input ↓ | Rerank gain |
|---|---:|---:|---:|---:|---:|
| `full` | **34.39** | **+15.42** | 33.47 | 46.57 | +0.04 |
| `natural_brio` | 32.20 | +13.22 | 30.75 | 45.55 | +0.10 |
| `natural_reverse` | 28.46 | +9.49 | 41.76 | 64.55 | +1.02 |
| `sep_baseline` (numeric-id prompt) | 24.04 | +5.06 | 43.37 | 71.18 | +1.70 |
| `natural_copy` | 19.95 | +0.98 | 44.92 | 76.90 | +3.34 |
| `natural_language` (baseline) | 18.98 | — | 45.16 | 78.15 | +3.60 |
| `natural_segments` | 18.73 | −0.25 | 44.50 | 78.12 | +4.24 |

![Training curves](assets/training_curves.png)

**What the numbers say**

- **BRIO is the single biggest win (+13.2).** It is the only component that optimises the shape of the metric directly. Once it switches on, copying drops from ~84 to ~45 BLEU vs input within a few epochs, and the graded score more than doubles.
- **Reverse augmentation is the best value (+9.5).** It is a pure data trick with no extra training cost.
- **The copy penalty barely helps on its own (+1.0).** It punishes single tokens, but copying is a sentence-level habit that cross-entropy keeps rewarding.
- **Reranking is a patch, not a fix.** It gives +4.2 to the weakest model and +0.04 to `full`, because once training has removed copies from the beams there is nothing left to rescue.
- **The gains overlap rather than stack.** The individual gains add up to +23.4, but `full` reaches +15.4, because every component attacks the same failure mode.
- **The prompt-format comparison is confounded.** `sep_baseline` beats `natural_language`, but it also differs in label smoothing (0.0 vs 0.1). The evidence points at label smoothing, not prompt wording.

### Honest caveat: the metric can be gamed

Penalized BLEU can reward broken text. A sentence with its first half cut off is far from the input and still shares n-grams with the reference. The two BRIO runs have the highest scores but also the lowest reference BLEU, and in my dev-set inspection **over half of `full`'s outputs started lowercase**, a sign of fragments. That is why the degenerate-output filter exists. If you want the best result that stays reliably well-formed, use **`natural_reverse` at 28.46 (+9.5)**. A next step would be to put the validity constraint inside the BRIO reward during training rather than only in the filter.

The submitted test predictions ([`predictions/`](predictions/etpc-paraphrase-generation-test-output.csv), 585 rows) contain no exact copies of the input, and 0.2% of them start lowercase.

## Quickstart

```bash
pip install -r requirements.txt          # Python 3.10+; install the CUDA build of torch for training
pytest -q                                # fast CPU checks, no model download
python bart_generation.py --run_name full                 # train + evaluate + write test predictions (GPU)
python bart_generation.py --run_name natural_reverse --no_segments --copy_weight 0 --brio_weight 0 --no_submission
python collect_metrics.py                # rebuild the ablation table from bart_generation_runs/*/metrics.json
python plot_results.py                   # rebuild the figures in assets/
python predict.py --checkpoint bart_generation_runs/full  # test predictions from a trained checkpoint
```

The trained checkpoints (~1.6 GB each) are not in the repo. Only their `metrics.json` files are kept, so `predict.py` needs a retrained or downloaded checkpoint.

<details><summary>Flags for each ablation run</summary>

| Run | Flags |
|---|---|
| `natural_language` | `--no_reverse --no_segments --copy_weight 0 --brio_weight 0` |
| `natural_segments` | `--no_reverse --copy_weight 0 --brio_weight 0` |
| `natural_copy` | `--no_reverse --no_segments --brio_weight 0` |
| `natural_reverse` | `--no_segments --copy_weight 0 --brio_weight 0` |
| `natural_brio` | `--no_reverse --no_segments --copy_weight 0` |
| `sep_baseline` | `--prompt sep --no_reverse --copy_weight 0 --brio_weight 0 --label_smoothing 0` |
| `full` | *(defaults)* |

Add `--no_submission` to every ablation run so it doesn't overwrite the test predictions.
</details>

## Hyperparameters

`facebook/bart-large` · AdamW lr 3e-5 (eps 1e-6, no weight decay) · batch 16 · linear warmup (10%), then decay over 12 epochs · max 20 epochs, patience 4 · label smoothing 0.1 · copy weight 0.5 · BRIO weight 0.2 from epoch 3 · 5 beams / 4 candidates · no-repeat 3-gram · 60 new tokens · seed 11711.

The learning rate reaches zero at epoch 12, so later epochs leave the weights unchanged; that is why the curves go flat.

## Repository layout

```
bart_generation.py      training, losses, BRIO, reranking, evaluation
predict.py              test predictions from a trained checkpoint
collect_metrics.py      ablation table from bart_generation_runs/*/metrics.json
plot_results.py         README figures
create_dev_split.py     80/20 split of the ETPC train set (seed 11711)
data/                   ETPC splits + type-id → name map
bart_generation_runs/   metrics.json per ablation run (+ ablation_metrics.csv)
predictions/            test-set output
tests/                  CPU smoke tests
```

## Credits

This was my individual task (Task 5) in the **Attention Seekers** team project for *Deep Learning for Natural Language Processing* (University of Göttingen, SS 2026). Teammates Irem Yurdakul, Vaishnavi Lalwani, Raj Randive and Burak Çolak worked on the other four tasks, which are not included here. The project scaffold is course-provided and Apache-2.0 licensed (see [LICENSE](LICENSE)).

**Data:** ETPC, Kovatchev, Martí & Salamó (2018), *ETPC: A paraphrase identification corpus annotated with extended paraphrase typology and negation*, LREC. Built on MRPC, Dolan & Brockett (2005).

**References**
- Lewis et al. (2020). BART: Denoising sequence-to-sequence pre-training. *ACL*.
- Liu, Liu, Radev & Neubig (2022). BRIO: Bringing order to abstractive summarization. *ACL*.
- Wahle, Gipp & Ruas (2023). Paraphrase types for generation and detection. *EMNLP*.
- Welleck et al. (2020). Neural text generation with unlikelihood training. *ICLR*.
- Szegedy et al. (2016). Rethinking the Inception architecture (label smoothing). *CVPR*.
- Post (2018). A call for clarity in reporting BLEU scores. *WMT*.
