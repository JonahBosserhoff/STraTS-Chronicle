"""One-off: overall evaluation (as overall_evaluation.py) of every ablation step in an ablation output folder.

    cd src && python evaluate_ablation.py --cohort <cohort_dir> [--ablation_dir ../outputs/chronicle_ablation]

Reads   <ablation_dir>/<run folder>/mask<p>_draw<d>/predictions_test_all_days.csv   (written by ablation.py)
        <cohort_dir>/<name of cohort_dir>.csv                                     sex per patient
Writes  <ablation_dir>/ablation_overall_evaluation.csv

Every label is scored with the functions of overall_evaluation.py, unchanged: C-index (OS, TTNT), AUROC with Chronicle's
thresholds (6 labs), AUROC (transfusion, febrile neutropenia); one random day per patient, 50 draws, seed 42. The
values compare with outputs/chronicle/overall_evaluation.csv of the training runs (= 0 % masked).
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from overall_evaluation import NOT_SCORED, RUN_DIR_PATTERN, label_tasks, load_sex, predictions_path

STEP_PATTERN = re.compile(r"^mask(?P<percentage>[0-9.]+)_draw(?P<mask_draw>\d+)$")
OUTPUT_NAME = "ablation_overall_evaluation.csv"


def main(args):
    ablation_dir = Path(args.ablation_dir)
    if not ablation_dir.is_dir():
        raise SystemExit(f"{ablation_dir} does not exist. Check --ablation_dir.")
    tasks = {target: (label, metric, definition, score)
             for label, target, metric, definition, score in label_tasks(load_sex(args.cohort))}
    label_order = {target: i for i, target in enumerate(tasks)}

    rows, unknown = [], set()
    for run_dir in sorted(p for p in ablation_dir.iterdir() if p.is_dir()):
        match = RUN_DIR_PATTERN.search(run_dir.name)
        if not match:
            continue
        target, run = match["target"], match["run"]
        if target not in tasks:
            unknown.add(target)
            continue
        label, metric, definition, score = tasks[target]
        for step_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            step = STEP_PATTERN.match(step_dir.name)
            if not step:
                continue
            percentage = float(step["percentage"])
            percentage = int(percentage) if percentage.is_integer() else percentage
            print(f"{label}: run {run}, {percentage} % masked, mask draw {step['mask_draw']}", flush=True)
            path = predictions_path(step_dir)
            result = dict(NOT_SCORED) if path is None else score(path)
            rows.append({"label": label, "target": target, "metric": metric, "definition": definition, "run": run,
                         "percentage": percentage, "mask_draw": int(step["mask_draw"]),
                         "predictions": path.name if path is not None else None, "run_dir": run_dir.name, **result})
    if unknown:
        print(f"⚠️ Not evaluated, unknown target: {', '.join(sorted(unknown))}")
    if not rows:
        raise SystemExit(f"{ablation_dir}: no ablation steps (<run folder>/mask<p>_draw<d>/) found.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["target", "run", "percentage", "mask_draw"], key=lambda col: col.map(label_order)
                        if col.name == "target" else col, kind="stable").reset_index(drop=True)
    out_path = ablation_dir / OUTPUT_NAME
    df.to_csv(out_path, index=False)

    overview = df.pivot_table(index="label", columns="percentage", values="value", aggfunc="mean")
    overview = overview.reindex([label for label in df["label"].unique() if label in overview.index])
    print("\nMean over runs and mask draws:\n")
    print(overview.round(3).to_string())
    not_scored = df.loc[df["value"].isna(), ["label", "run", "percentage", "mask_draw"]]
    if len(not_scored):
        print(f"\nWARNING: {len(not_scored)}/{len(df)} rows not scored (no predictions, scaled units, no events):")
        print(not_scored.to_string(index=False))
    print(f"\nWrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overall evaluation of every step of a feature ablation.")
    p.add_argument("--ablation_dir", type=str, default="../outputs/chronicle_ablation", help="output_dir of ablation.py")
    p.add_argument("--cohort", type=str, required=True, help="cohort directory holding <name of the directory>.csv")
    main(p.parse_args())
