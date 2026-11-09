"""Overall evaluation of all STraTS Chronicle runs in one folder: C-index (OS, TTNT) and AUROC (6 labs, transfusion,
febrile neutropenia). Same thresholds, sampling and seed as Chronicle's overall_evaluation.py (and evaluate_os.py of
the single-target repo), so the values compare.

    cd src && python overall_evaluation.py --cohort <cohort_dir> [--results_dir ../outputs/chronicle]

Reads   <results_dir>/<run>/predictions_test_all_days.csv   every run folder of main.py --dataset chronicle; the
                                                            target comes from the folder name (|target:<T>|)
        <cohort_dir>/<name of cohort_dir>.csv               sex per patient (feature_name == "gender_female")
Writes  <results_dir>/overall_evaluation.csv

predictions_test_all_days.csv (main.py, evaluator_survival.py) holds every test day with a target, one row per
patient and day:
    survival     patient_id, day, prediction, time, event      prediction: risk score, higher = shorter time
    regression   patient_id, day, prediction, ground_truth     original units (Chronicle label scaler), ground truth
                                                               binarised with LAB_LABELS
    binary       patient_id, day, prediction, ground_truth     ground truth 0/1, prediction = logit
Runs from before main.py saved all days for survival targets only have predictions_test.csv (the num_eval_draws
drawn days per patient): their distinct (patient, day) rows are used instead, with a warning. These are fewer days
than Chronicle samples from, so the value is not quite comparable.

A regression run whose log says its predictions were saved in the scaled units (no Chronicle label scaler) is not
scored: the thresholds are in the original units.
Rows of a sex-specific lab (AST, ALT, creatinine) of patients with unknown sex are left out, with a warning.

Every metric is computed on one random row per patient, N_ITER times. The interval is the 2.5 / 97.5 percentile over
these draws: it reflects which day was drawn per patient, not the sampling of the patients.

This is not the same number as "Test res of the best checkpoint" in log.txt: that one is the mean over
--num_eval_draws (10) draws of cut days fixed with --seed (2023), this one over N_ITER (50) draws with SEED (42).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

ALL_DAYS_NAME = "predictions_test_all_days.csv"
DRAWS_NAME = "predictions_test.csv"  # fallback, see the docstring
OUTPUT_NAME = "overall_evaluation.csv"
SCALED_UNITS_WARNING = "predictions are saved in the scaled units"  # main.py, no Chronicle label scaler
N_ITER = 50
SEED = 42

SURVIVAL_LABELS = ["OS", "TTNT"]

# lab name -> (target, threshold, direction); same thresholds as Chronicle. Tuples: (male, female).
# direction "min": event = ground_truth < threshold, a low prediction means event
# direction "max": event = ground_truth > threshold, a high prediction means event
LAB_LABELS = {
    "hemoglobin": ("label_min_lab_hemoglobin_60_days", 8.0, "min"),
    "neutrophil_granulocytes_absolute": ("label_min_lab_neutrophil_granulocytes_absolute_60_days", 1.0, "min"),
    "platelets": ("label_min_lab_platelets_60_days", 50.0, "min"),
    "ast": ("label_max_lab_ast_60_days", (150.0, 105.0), "max"),
    "alt": ("label_max_lab_alt_60_days", (150.0, 105.0), "max"),
    "creatinine": ("label_max_lab_creatinine_60_days", (1.95, 1.65), "max"),
}

BINARY_LABELS = {
    "transfusion": "label_supportive_care_transfusion_7_days",
    "febrile_neutropenia": "label_febrile_neutropenia_21d",
}

SURVIVAL_COLS = ["patient_id", "day", "time", "event", "prediction"]
LAB_COLS = ["patient_id", "day", "prediction", "ground_truth"]

RUN_DIR_PATTERN = re.compile(r"\|target:(?P<target>.+?)\|train_frac:.*\|run:(?P<run>[^|]+)$")


def patient_id_to_int(pid):
    """The integer id of a patient, as in the single-target repo (src/chronos/dataset_os.py), so that the ids of the
    predictions and of the cohort CSV match however pandas parsed them."""
    try:
        return int(pid)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(pid))
        if not match:
            raise ValueError(f"Cannot derive a numeric id from patient_id {pid!r}.") from None
        return int(match.group())


def load_sex(cohort):
    """patient_id (as int) -> sex (1 = female, 0 = male) from the raw long-format cohort CSV."""
    path = Path(cohort) / f"{Path(cohort).name}.csv"
    if not path.is_file():
        raise SystemExit(f"Cohort CSV not found: {path}. Check --cohort.")
    df = pd.read_csv(path, usecols=["patient_id", "feature_name", "feature_value"], low_memory=False)
    rows = df.loc[df["feature_name"] == "gender_female", ["patient_id", "feature_value"]].dropna()
    if rows.empty:
        raise SystemExit(f"{path}: no rows with feature_name == 'gender_female'.")
    rows = rows.drop_duplicates("patient_id")
    sex = pd.Series(
        rows["feature_value"].astype(float).to_numpy(),
        index=[patient_id_to_int(pid) for pid in rows["patient_id"]],
    )
    if sex.index.has_duplicates:
        raise SystemExit(f"{path}: different patient_ids map to the same numeric id, sex is ambiguous.")
    return sex


def find_runs(results_dir):
    """(target, run tag, run folder) of every run folder of main.py --dataset chronicle, sorted by folder name."""
    runs = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        match = RUN_DIR_PATTERN.search(run_dir.name)
        if match:
            runs.append((match["target"], match["run"], run_dir))
    return runs


def predictions_path(run_dir):
    """The predictions of every test day with a target, or the drawn days of an older survival run (with a
    warning), or None."""
    if (run_dir / ALL_DAYS_NAME).is_file():
        return run_dir / ALL_DAYS_NAME
    if (run_dir / DRAWS_NAME).is_file():
        print(f"⚠️ {run_dir.name}: no {ALL_DAYS_NAME} (run from before it was saved for every target), "
              f"using the drawn days of {DRAWS_NAME}: fewer days per patient than Chronicle samples from.")  # fmt: skip
        return run_dir / DRAWS_NAME
    print(f"⚠️ {run_dir.name}: no predictions (run not finished?), not evaluated.")
    return None


def load_predictions(path, columns):
    """Rows of one predictions file without NaNs in ``columns``, one row per patient and day."""
    df = pd.read_csv(path, usecols=columns)
    df = df.drop_duplicates(["patient_id", "day"]).reset_index(drop=True)  # draws of predictions_test.csv
    na_mask = df.isna().any(axis=1)
    if na_mask.any():
        print(f"⚠️ {path}: dropped {int(na_mask.sum())}/{len(df)} rows with NaNs.")
        df = df.loc[~na_mask].reset_index(drop=True)
    return df


def saved_in_scaled_units(run_dir):
    log = run_dir / "log.txt"
    return log.is_file() and SCALED_UNITS_WARNING in log.read_text(errors="replace")


def with_row_threshold(df, path, threshold, sex):
    """``df`` with the event threshold of each row; for a (male, female) threshold, known sex only."""
    if not isinstance(threshold, tuple):
        return df.assign(threshold=threshold)
    per_row = df["patient_id"].map(patient_id_to_int).map(sex).map({1.0: threshold[1], 0.0: threshold[0]})
    unknown = per_row.isna()
    if unknown.any():
        print(
            f"⚠️ {path}: dropped {int(unknown.sum())}/{len(df)} rows of "
            f"{df.loc[unknown, 'patient_id'].nunique()} patients with unknown sex."
        )
    return df.assign(threshold=per_row).loc[~unknown].reset_index(drop=True)


def _summarise(scores, event_counts, n_patients):
    """The result row of one label: mean, std and interval of the finite scores (NaNs if there is none)."""
    finite = scores[~np.isnan(scores)]
    empty = finite.size == 0
    return {
        "value": np.nan if empty else float(finite.mean()),
        "std": np.nan if empty else float(finite.std()),
        "ci_lo": np.nan if empty else float(np.percentile(finite, 2.5)),
        "ci_hi": np.nan if empty else float(np.percentile(finite, 97.5)),
        "n_patients": n_patients,
        "n_events_mean": float(event_counts.mean()),
        "prevalence_mean": float(event_counts.mean() / n_patients),
        "n_iter": scores.size,
    }


NOT_SCORED = {"value": np.nan, "std": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
              "n_patients": 0, "n_events_mean": np.nan, "prevalence_mean": np.nan, "n_iter": 0}  # fmt: skip


def cindex_sampled(df, n_iter=N_ITER, seed=SEED):
    """C-index of the risk score (higher = shorter time to the event), one random row per patient x n_iter."""
    if df.empty:
        return dict(NOT_SCORED)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_iter)
    event_counts = np.empty(n_iter)
    grouped = df.groupby("patient_id", sort=False)
    for i in range(n_iter):
        subset = grouped.sample(n=1, random_state=rng.integers(0, 2**32 - 1))
        event_counts[i] = subset["event"].sum()
        try:
            scores[i] = concordance_index(subset["time"], -subset["prediction"], subset["event"])
        except ZeroDivisionError:  # no admissible pairs
            scores[i] = np.nan
    return _summarise(scores, event_counts, df["patient_id"].nunique())


def auroc_sampled(df, y_true, y_score, n_iter=N_ITER, seed=SEED):
    """AUROC, one random row per patient x n_iter. ``y_true`` / ``y_score``: 0/1 labels / scores of the rows of df."""
    if df.empty:
        return dict(NOT_SCORED)
    rows_by_patient = list(df.groupby("patient_id", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    scores = np.empty(n_iter)
    event_counts = np.empty(n_iter)
    for i in range(n_iter):
        picks = np.array([rows[rng.integers(len(rows))] for rows in rows_by_patient])
        yt = y_true[picks]
        event_counts[i] = yt.sum()
        scores[i] = np.nan if yt.sum() in (0, yt.size) else roc_auc_score(yt, y_score[picks])
    return _summarise(scores, event_counts, len(rows_by_patient))


def score_survival(path):
    return cindex_sampled(load_predictions(path, SURVIVAL_COLS))


def score_lab(path, threshold, direction, sex):
    if saved_in_scaled_units(path.parent):
        print(f"⚠️ {path}: predictions in the scaled units (no Chronicle label scaler), not evaluated.")
        return dict(NOT_SCORED)
    df = with_row_threshold(load_predictions(path, LAB_COLS), path, threshold, sex)
    truth, cut, prediction = df["ground_truth"].to_numpy(), df["threshold"].to_numpy(), df["prediction"].to_numpy()
    if direction == "min":
        return auroc_sampled(df, (truth < cut).astype(int), -prediction)
    return auroc_sampled(df, (truth > cut).astype(int), prediction)


def score_binary(path):
    df = load_predictions(path, LAB_COLS)
    return auroc_sampled(df, df["ground_truth"].to_numpy().astype(int), df["prediction"].to_numpy())


def lab_definition(threshold, direction):
    op = "<" if direction == "min" else ">"
    if isinstance(threshold, tuple):
        return f"{op} {threshold[0]:g} (m) / {threshold[1]:g} (f)"
    return f"{op} {threshold:g}"


def label_tasks(sex):
    """(label, target, metric, definition, path -> result row) of every label, in reporting order."""
    tasks = [(label, label, "C-index", "time to event", score_survival) for label in SURVIVAL_LABELS]
    for lab, (target, threshold, direction) in LAB_LABELS.items():
        def score(path, threshold=threshold, direction=direction):
            return score_lab(path, threshold, direction, sex)

        tasks.append((lab, target, "AUROC", lab_definition(threshold, direction), score))
    tasks += [(lab, target, "AUROC", "0/1", score_binary) for lab, target in BINARY_LABELS.items()]
    return tasks


def overview_row(row):
    mean = row["value"]
    return {
        "label": row["label"],
        "run": row["run"],
        "definition": row["definition"],
        "metric": row["metric"],
        "value [interval]": "—" if np.isnan(mean) else f"{mean:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
        "patients": row["n_patients"],
        "events": "—" if np.isnan(row["n_events_mean"]) else f"{row['n_events_mean']:.0f}",
        "prevalence": "—" if np.isnan(row["prevalence_mean"]) else f"{row['prevalence_mean']:.1%}",
    }


def main(args):
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"{results_dir} does not exist. Check --results_dir.")
    sex = load_sex(args.cohort)
    runs = find_runs(results_dir)
    if not runs:
        raise SystemExit(f"{results_dir}: no run folders (…|target:<T>|train_frac:…|run:<r>). Check --results_dir.")

    tasks = label_tasks(sex)
    rows = []
    for label, target, metric, definition, score in tasks:
        info = {"label": label, "metric": metric, "definition": definition, "target": target}
        target_runs = [(run, run_dir) for t, run, run_dir in runs if t == target]
        if not target_runs:
            rows.append({**info, "run": "—", "run_dir": "—", **NOT_SCORED})
        for run, run_dir in target_runs:
            print(f"{label}: run {run}", flush=True)
            path = predictions_path(run_dir)
            result = dict(NOT_SCORED) if path is None else score(path)
            rows.append({**info, "run": run, "run_dir": run_dir.name, "predictions": path and path.name, **result})

    known = {target for _, target, *_ in tasks}
    unknown = sorted({target for target, *_ in runs if target not in known})
    if unknown:
        print(f"⚠️ Not evaluated, unknown target: {', '.join(unknown)}")

    out_path = results_dir / OUTPUT_NAME
    pd.DataFrame(rows).to_csv(out_path, index=False)

    overview = pd.DataFrame([overview_row(row) for row in rows])
    print(f"\nResults: {results_dir}\n")
    print(overview.to_string(index=False))

    not_scored = overview.loc[overview["value [interval]"] == "—", "label"].tolist()
    if not_scored:
        print(f"\nWARNING: {len(not_scored)}/{len(overview)} rows not scored (no run / predictions, scaled units, "
              f"sex unknown, or no events): {', '.join(not_scored)}")  # fmt: skip
    print(f"\nWrote {out_path}")


def build_parser():
    p = argparse.ArgumentParser(description="C-index / AUROC of all STraTS Chronicle runs in one folder.")
    p.add_argument("--results_dir", type=str, default="../outputs/chronicle", help="folder holding one subfolder per run")
    p.add_argument("--cohort", type=str, required=True, help="cohort directory holding <name of the directory>.csv")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
