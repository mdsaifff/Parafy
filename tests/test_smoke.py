"""Fast CPU checks of the pure-Python pieces; no model download needed."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bart_generation as bg  # noqa: E402


def test_degenerate_outputs_are_rejected():
    assert bg.is_degenerate("")
    assert bg.is_degenerate(" started mid sentence.")
    assert bg.is_degenerate("shares fell sharply.")                # lowercase start = cut-off fragment
    assert bg.is_degenerate("It was in in in the house.")
    assert not bg.is_degenerate("Kraft shares fell 25 cents.")


def test_reverse_augment_swaps_pairs_but_skips_entailment():
    df = pd.DataFrame({"id": ["a", "b"], "sentence1": ["A one.", "B one."], "sentence2": ["A two.", "B two."],
                       "paraphrase_type_ids": ["[5, 29]", "[31]"],
                       bg.SEGMENT_COLUMN: ["[1]", "[2]"], "sentence2_segment_location": ["[3]", "[4]"]})
    out = bg.reverse_augment(df)
    assert len(out) == 3                                          # entailment row (31) is one-directional
    swapped = out[out["id"] == "a-reverse"].iloc[0]
    assert (swapped.sentence1, swapped.sentence2, swapped[bg.SEGMENT_COLUMN]) == ("A two.", "A one.", "[3]")


def test_leaky_rows_are_removed():
    train = pd.DataFrame({"id": ["x", "y"], "sentence1": ["Same  sentence.", "Other."], "sentence2": ["Q.", "R."]})
    dev = pd.DataFrame({"id": ["z"], "sentence1": ["same sentence."], "sentence2": ["S."]})
    assert bg.remove_leaky_training_rows(train, dev)["id"].tolist() == ["y"]


def test_natural_prompt_uses_type_names_and_drops_identity():
    row = next(pd.DataFrame({"sentence1": ["Hi there."], "paraphrase_type_ids": ["[29, 5]"],
                             bg.SEGMENT_COLUMN: [None]}).itertuples())
    prompt = bg.make_input(row)
    assert "Sentence: Hi there." in prompt and bg.ETPC_TYPE_NAMES[5] in prompt
    assert bg.ETPC_TYPE_NAMES[29] not in prompt


def test_reranker_avoids_exact_copy_and_fragments():
    source = "The company reported higher profits this quarter."
    candidates = [source, "profits rose this quarter.", "This quarter, the firm posted bigger profits."]
    assert bg.select_candidate(source, "Same Polarity Substitution (contextual)", candidates,
                               [-0.1, -0.2, -0.3]) == candidates[2]
