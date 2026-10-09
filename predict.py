"""Write the ETPC test predictions from a trained checkpoint, without retraining.

    python predict.py                                   # uses bart_generation_runs/full
    python predict.py --checkpoint <local dir or Hugging Face Hub id>
"""

import argparse
import os
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from bart_generation import ROOT, RUNS_DIR, SUBMISSION_PATH, test_model, transform_data

# the decoding settings bart_generation_runs/full was trained with, copied from its metrics.json
ARGS = SimpleNamespace(batch_size=16, max_length=256, num_beams=5, num_candidates=4,
                       max_new_tokens=60, length_penalty=1.0, no_repeat_ngram_size=3)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", default=os.path.join(RUNS_DIR, "full"))
parser.add_argument("--out", default=SUBMISSION_PATH)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = BartForConditionalGeneration.from_pretrained(args.checkpoint).to(device)

test_df = pd.read_csv(os.path.join(ROOT, "data", "etpc-paraphrase-generation-test-student.csv"))
test_data = transform_data(test_df, tokenizer, ARGS.max_length, ARGS.batch_size, False, "natural", True)
test_output = test_model(model, test_data, test_df, device, tokenizer, ARGS)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
test_output.to_csv(args.out, index=False)
print(f"Wrote {args.out} with {len(test_output)} rows.")
