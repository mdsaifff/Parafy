"""Aggregate the per-run metrics.json files into one ablation table.

Each bart_generation.py run drops bart_generation_runs/<run_name>/metrics.json at the end. This
reads them all, lines the runs up against a baseline cell and reports what each
improvement is worth in penalized BLEU. Runs that died before writing their json
can still be recovered from the slurm logs with --logs.
"""

import argparse
import csv
import glob
import json
import os
import re

# "epoch 3: BLEU 33.385 | negative input BLEU 51.378 | penalized BLEU 31.216"
EPOCH_LINE = re.compile(r"^(.+?): BLEU ([\d.]+) \| negative input BLEU ([\d.]+) \| penalized BLEU ([\d.]+)")
RUN_LINE = re.compile(r"^Run (\S+) on ")


def load_run(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    name = data.get("run_name") or data.get("experiment") or os.path.basename(os.path.dirname(path))
    rerank = data.get("dev_rerank") or data           # old flat format keeps the scores at top level
    top_beam = data.get("dev_top_beam") or {}
    config = data.get("config", data)
    epochs = data.get("epochs", [])
    return {
        "run": name,
        "source": "metrics.json",
        "best_penalized_bleu": data.get("best_epoch_penalized_bleu",
                                        max((e["penalized_bleu"] for e in epochs), default=rerank.get("penalized_bleu"))),
        "penalized_bleu": rerank.get("penalized_bleu"),
        "corpus_bleu": rerank.get("corpus_bleu"),
        "input_bleu": rerank.get("input_bleu"),
        "top_beam_penalized_bleu": top_beam.get("penalized_bleu"),
        "rerank_delta": data.get("rerank_delta"),
        "epochs_run": len(epochs) or None,
        "prompt": config.get("prompt"),
        "segments": config.get("segments"),
        "reverse": config.get("reverse"),
        "label_smoothing": config.get("label_smoothing"),
        "copy_weight": config.get("copy_weight"),
        "brio_weight": config.get("brio_weight", config.get("brio")),
        "seed": data.get("seed"),
    }


def load_log(path):
    """recover a killed run from its slurm output: best epoch seen before the job died"""
    name, best, last = None, None, None
    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            run = RUN_LINE.match(line)
            if run:
                name = run.group(1)
                continue
            hit = EPOCH_LINE.match(line.strip())
            if hit and hit.group(1).startswith("epoch"):
                scores = {"corpus_bleu": float(hit.group(2)), "input_bleu": float(hit.group(3)),
                          "penalized_bleu": float(hit.group(4))}
                last = scores
                if best is None or scores["penalized_bleu"] > best["penalized_bleu"]:
                    best = scores
    if name is None or best is None:
        return None
    return {"run": name, "source": os.path.basename(path), "best_penalized_bleu": best["penalized_bleu"],
            "penalized_bleu": last["penalized_bleu"], "corpus_bleu": last["corpus_bleu"],
            "input_bleu": last["input_bleu"], "top_beam_penalized_bleu": None, "rerank_delta": None,
            "epochs_run": None, "prompt": None, "segments": None, "reverse": None,
            "label_smoothing": None, "copy_weight": None, "brio_weight": None, "seed": None}


def cell(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description="collect ablation metrics into one table")
    root = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--runs_dir", default=os.path.join(root, "bart_generation_runs"))
    parser.add_argument("--logs", default=None, help="slurm output directory, for runs with no metrics.json")
    parser.add_argument("--baseline", default="natural_language", help="run the deltas are measured against")
    parser.add_argument("--out", default=os.path.join(root, "bart_generation_runs", "ablation_metrics.csv"))
    args = parser.parse_args()

    runs = {}
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "**", "*metrics.json"), recursive=True)):
        # skip archived grids. bart_generation_runs/_archive_first_grid/ keeps an older json format whose
        # rows come out empty and whose run names collide with the current grid
        if any(part.startswith("_") for part in os.path.normpath(path).split(os.sep)):
            continue
        row = load_run(path)
        runs[row["run"]] = row
    if args.logs:
        for path in sorted(glob.glob(os.path.join(args.logs, "*.out"))):
            row = load_log(path)
            if row and row["run"] not in runs:      # a real metrics.json always wins
                runs[row["run"]] = row

    if not runs:
        print(f"No metrics found under {args.runs_dir}.")
        return

    base = runs.get(args.baseline)
    if base is None:
        print(f"Baseline {args.baseline!r} not found, reporting absolute scores only.")
    ordered = sorted(runs.values(), key=lambda r: -(r["best_penalized_bleu"] or 0))
    for row in ordered:
        row["delta_vs_baseline"] = (None if base is None or row["best_penalized_bleu"] is None
                                    else row["best_penalized_bleu"] - base["best_penalized_bleu"])

    fields = ["run", "best_penalized_bleu", "penalized_bleu", "corpus_bleu", "input_bleu",
              "top_beam_penalized_bleu", "rerank_delta", "delta_vs_baseline", "epochs_run",
              "prompt", "segments", "reverse", "label_smoothing", "copy_weight", "brio_weight",
              "seed", "source"]
    with open(args.out, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    header = ["Run", "Best penalized BLEU", "BLEU", "Neg. input BLEU", "Rerank delta", f"vs {args.baseline}"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in ordered:
        print("| " + " | ".join([row["run"], cell(row["best_penalized_bleu"]), cell(row["corpus_bleu"]),
                                 cell(row["input_bleu"]), cell(row["rerank_delta"]),
                                 "-" if row["delta_vs_baseline"] is None else f"{row['delta_vs_baseline']:+.3f}"]) + " |")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
