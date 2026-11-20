"""Overall evaluation of all STraTS Chronicle runs in one folder: C-index (OS, TTNT) and AUROC (6 labs, transfusion,
febrile neutropenia), with a 95% patient-bootstrap CI. Same thresholds, sampling, bootstrap and seed as Chronicle's
overall_evaluation.py (and evaluate_os.py of the single-target repo), so the values compare.

    cd src && python overall_evaluation.py --cohort <cohort_dir> [--results_dir ../outputs/chronicle]
    cd src && python overall_evaluation.py --external --cohort <external_cohort_dir> \\
                                           [--results_dir ../outputs/chronicle_external]

Reads   <results_dir>/<run>/predictions_test_all_days.csv   every run folder of main.py --dataset chronicle; the
                                                            target comes from the folder name (|target:<T>|)
        <cohort_dir>/<name of cohort_dir>.csv               sex per patient (feature_name == "gender_female")
Writes  <results_dir>/overall_evaluation.csv, <results_dir>/bootstrap_replicates.csv (one value per replicate)

--external: the output_dir of external_validation.py instead (results_dir itself and every subfolder of it with
predictions_external_all_days.csv). The target and run come from the folder name if it is a run folder name,
otherwise from model_dir in its log.txt (the last one, if the log holds several). One output_dir per trained model:
external_validation.py overwrites the predictions in output_dir. --cohort: the raw CSV of the external cohort.

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

Point estimate: every metric is computed on one random row per patient, N_ITER times; the mean over these draws.
ci_lo / ci_hi: 2.5 / 97.5 percentile over these draws; they reflect which day was drawn per patient, not the
sampling of the patients.

Bootstrap (boot_ci_lo / boot_ci_hi): N_BOOTSTRAP (500) replicates. Each replicate resamples the patients with
replacement and then, like the point estimate, draws one random row for every drawn patient N_ITER (50) times (a
patient drawn twice counts as two patients, each with its own random day), i.e. 500 x 50 = 25000 metric values. The
replicate value is the mean of the metric over its 50 draws, so the interval is that of the point estimate (the mean
over the day draws) under the sampling of the patients: 2.5 / 97.5 percentile over the 500 replicates. Day draws
without a defined metric (one class only, no admissible pairs) are left out of a replicate's mean, replicates
without any defined draw out of the interval, both with a warning. The point estimate and ci_lo / ci_hi are
unchanged by the bootstrap (own random generator). --n_boot changes the number of replicates, 0 skips the bootstrap.

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
EXTERNAL_ALL_DAYS_NAME = "predictions_external_all_days.csv"  # external_validation.py
EXTERNAL_DRAWS_NAME = "predictions_external.csv"
OUTPUT_NAME = "overall_evaluation.csv"
REPLICATES_NAME = "bootstrap_replicates.csv"
SCALED_UNITS_WARNING = "predictions are saved in the scaled units"  # main.py, no Chronicle label scaler
EXTERNAL_SCALED_UNITS_WARNING = "predictions are saved in the units of the model output"  # external_validation.py
N_ITER = 50
N_BOOTSTRAP = 500
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
# model_dir of the training run in the log of external_validation.py (str of its config)
EXTERNAL_LOG_PATTERN = re.compile(
    r"'model_dir': '[^']*\|target:(?P<target>[^|']+)\|train_frac:[^|']*\|run:(?P<run>[^|'/]+)/?'"
)


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


def find_external_runs(results_dir):
    """(target, run tag, folder) of results_dir and every subfolder with predictions of external_validation.py; the
    target from the folder name or else from model_dir in its log.txt."""
    runs = []
    for run_dir in [results_dir, *sorted(p for p in results_dir.iterdir() if p.is_dir())]:
        if not any((run_dir / name).is_file() for name in (EXTERNAL_ALL_DAYS_NAME, EXTERNAL_DRAWS_NAME)):
            continue
        match = RUN_DIR_PATTERN.search(run_dir.name)
        if match is None and (run_dir / "log.txt").is_file():
            matches = list(EXTERNAL_LOG_PATTERN.finditer((run_dir / "log.txt").read_text(errors="replace")))
            match = matches[-1] if matches else None
        if match is None:
            print(f"⚠️ {run_dir}: predictions, but no target (no run folder name, no model_dir in log.txt), skipped.")
            continue
        runs.append((match["target"], match["run"], run_dir))
    return runs


def predictions_path(run_dir, all_days_name=ALL_DAYS_NAME, draws_name=DRAWS_NAME):
    """The predictions of every day with a target, or the drawn days of an older survival run (with a warning), or
    None."""
    if (run_dir / all_days_name).is_file():
        return run_dir / all_days_name
    if (run_dir / draws_name).is_file():
        print(f"⚠️ {run_dir.name}: no {all_days_name} (run from before it was saved for every target), "
              f"using the drawn days of {draws_name}: fewer days per patient than Chronicle samples from.")  # fmt: skip
        return run_dir / draws_name
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
    if not log.is_file():
        return False
    text = log.read_text(errors="replace")
    return SCALED_UNITS_WARNING in text or EXTERNAL_SCALED_UNITS_WARNING in text


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


BOOT_NOT_SCORED = {"boot_ci_lo": np.nan, "boot_ci_hi": np.nan, "n_boot_valid": 0, "n_boot": 0}
NOT_SCORED = {"value": np.nan, "std": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
              "n_patients": 0, "n_events_mean": np.nan, "prevalence_mean": np.nan, "n_iter": 0,
              **BOOT_NOT_SCORED}  # fmt: skip
REPLICATES_KEY = "boot_replicates"  # values of the replicates in a result row; main() moves them to REPLICATES_NAME


def _cindex_safe(time, risk, event):
    """lifelines C-index of ``risk`` (higher = shorter time); NaN without admissible pairs."""
    try:
        return concordance_index(time, -risk, event)
    except ZeroDivisionError:  # no admissible pairs
        return np.nan


def _auroc_safe(y_true, y_score):
    if y_true.size == 0 or y_true.sum() in (0, y_true.size):
        return np.nan
    return roc_auc_score(y_true, y_score)


def bootstrap(pids, score_fn, label, n_iter=N_ITER, n_boot=N_BOOTSTRAP, seed=SEED):
    """Patient bootstrap: ``n_boot`` replicates. Each replicate resamples the patients with replacement, then draws
    one random row for every drawn patient ``n_iter`` times (copies independently); the replicate value is the mean
    of the metric over these ``n_iter`` draws (undefined draws left out), as the point estimate.
    ``score_fn(row positions) -> metric``. Returns the summary dict, with the replicate values under
    REPLICATES_KEY."""
    if n_boot <= 0 or len(pids) == 0:
        return dict(BOOT_NOT_SCORED)
    codes, uniques = pd.factorize(pids)
    n = len(uniques)
    order = np.argsort(codes, kind="stable")  # rows grouped by patient
    bounds = np.searchsorted(codes[order], np.arange(n + 1))  # rows of patient k: order[bounds[k]:bounds[k + 1]]

    rng = np.random.default_rng([seed, 1])  # separate from the day draws of the point estimate
    values = np.empty(n_boot)
    n_draws_undefined = 0
    for b in range(n_boot):
        draw = rng.integers(0, n, n)
        starts, lengths = bounds[draw], bounds[draw + 1] - bounds[draw]
        draws = np.empty(n_iter)
        for i in range(n_iter):
            picks = order[starts + (rng.random(n) * lengths).astype(np.int64)]
            draws[i] = score_fn(picks)
        undefined = np.isnan(draws)
        n_draws_undefined += int(undefined.sum())
        values[b] = draws[~undefined].mean() if not undefined.all() else np.nan

    if n_draws_undefined:
        print(f"⚠️ {label}: {n_draws_undefined}/{n_boot * n_iter} day draws within the bootstrap replicates "
              "undefined (one class only / no admissible pairs), left out of the replicate means.")  # fmt: skip
    valid = ~np.isnan(values)
    if not valid.all():
        print(f"⚠️ {label}: {int((~valid).sum())}/{n_boot} bootstrap replicates undefined (every day draw "
              "undefined), left out of the interval.")  # fmt: skip
    finite = values[valid]
    return {
        "boot_ci_lo": float(np.percentile(finite, 2.5)) if finite.size else np.nan,
        "boot_ci_hi": float(np.percentile(finite, 97.5)) if finite.size else np.nan,
        "n_boot_valid": int(valid.sum()),
        "n_boot": n_boot,
        REPLICATES_KEY: values,
    }


def cindex_sampled(df, n_iter=N_ITER, seed=SEED, n_boot=0, label="C-index"):
    """C-index of the risk score (higher = shorter time to the event), one random row per patient x n_iter, plus
    the patient bootstrap (n_boot replicates)."""
    if df.empty:
        return dict(NOT_SCORED)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_iter)
    event_counts = np.empty(n_iter)
    grouped = df.groupby("patient_id", sort=False)
    for i in range(n_iter):
        subset = grouped.sample(n=1, random_state=rng.integers(0, 2**32 - 1))
        event_counts[i] = subset["event"].sum()
        scores[i] = _cindex_safe(subset["time"], subset["prediction"], subset["event"])

    time, risk, event = df["time"].to_numpy(), df["prediction"].to_numpy(), df["event"].to_numpy()
    boot = bootstrap(df["patient_id"].to_numpy(), lambda p: _cindex_safe(time[p], risk[p], event[p]), label,
                     n_iter=n_iter, n_boot=n_boot, seed=seed)  # fmt: skip
    return {**_summarise(scores, event_counts, df["patient_id"].nunique()), **boot}


def auroc_sampled(df, y_true, y_score, n_iter=N_ITER, seed=SEED, n_boot=0, label="AUROC"):
    """AUROC, one random row per patient x n_iter, plus the patient bootstrap (n_boot replicates).
    ``y_true`` / ``y_score``: 0/1 labels / scores of the rows of df."""
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
        scores[i] = _auroc_safe(yt, y_score[picks])

    boot = bootstrap(df["patient_id"].to_numpy(), lambda p: _auroc_safe(y_true[p], y_score[p]), label,
                     n_iter=n_iter, n_boot=n_boot, seed=seed)  # fmt: skip
    return {**_summarise(scores, event_counts, len(rows_by_patient)), **boot}


def score_survival(path, n_boot=0, label="C-index"):
    return cindex_sampled(load_predictions(path, SURVIVAL_COLS), n_boot=n_boot, label=label)


def score_lab(path, threshold, direction, sex, n_boot=0, label="AUROC"):
    if saved_in_scaled_units(path.parent):
        print(f"⚠️ {path}: predictions in the scaled units (no Chronicle label scaler), not evaluated.")
        return dict(NOT_SCORED)
    df = with_row_threshold(load_predictions(path, LAB_COLS), path, threshold, sex)
    truth, cut, prediction = df["ground_truth"].to_numpy(), df["threshold"].to_numpy(), df["prediction"].to_numpy()
    if direction == "min":
        return auroc_sampled(df, (truth < cut).astype(int), -prediction, n_boot=n_boot, label=label)
    return auroc_sampled(df, (truth > cut).astype(int), prediction, n_boot=n_boot, label=label)


def score_binary(path, n_boot=0, label="AUROC"):
    df = load_predictions(path, LAB_COLS)
    return auroc_sampled(df, df["ground_truth"].to_numpy().astype(int), df["prediction"].to_numpy(),
                         n_boot=n_boot, label=label)  # fmt: skip


def lab_definition(threshold, direction):
    op = "<" if direction == "min" else ">"
    if isinstance(threshold, tuple):
        return f"{op} {threshold[0]:g} (m) / {threshold[1]:g} (f)"
    return f"{op} {threshold:g}"


def label_tasks(sex, n_boot=0):
    """(label, target, metric, definition, path -> result row) of every label, in reporting order. ``n_boot``:
    bootstrap replicates (0: none, as for the ablation); the result row then holds their values under
    REPLICATES_KEY."""
    tasks = []
    for label in SURVIVAL_LABELS:
        def score(path, label=label):
            return score_survival(path, n_boot, label)

        tasks.append((label, label, "C-index", "time to event", score))
    for lab, (target, threshold, direction) in LAB_LABELS.items():
        def score(path, lab=lab, threshold=threshold, direction=direction):
            return score_lab(path, threshold, direction, sex, n_boot, lab)

        tasks.append((lab, target, "AUROC", lab_definition(threshold, direction), score))
    for lab, target in BINARY_LABELS.items():
        def score(path, lab=lab):
            return score_binary(path, n_boot, lab)

        tasks.append((lab, target, "AUROC", "0/1", score))
    return tasks


def overview_row(row):
    mean, boot_lo = row["value"], row["boot_ci_lo"]
    return {
        "label": row["label"],
        "run": row["run"],
        "definition": row["definition"],
        "metric": row["metric"],
        "value [interval]": "—" if np.isnan(mean) else f"{mean:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
        "bootstrap 95% CI": "—" if np.isnan(boot_lo) else f"[{boot_lo:.3f}, {row['boot_ci_hi']:.3f}]",
        "patients": row["n_patients"],
        "events": "—" if np.isnan(row["n_events_mean"]) else f"{row['n_events_mean']:.0f}",
        "prevalence": "—" if np.isnan(row["prevalence_mean"]) else f"{row['prevalence_mean']:.1%}",
    }


def main(args):
    results_dir = Path(args.results_dir or ("../outputs/chronicle_external" if args.external else "../outputs/chronicle"))
    if not results_dir.is_dir():
        raise SystemExit(f"{results_dir} does not exist. Check --results_dir.")
    sex = load_sex(args.cohort)
    if args.external:
        runs = find_external_runs(results_dir)
        names = (EXTERNAL_ALL_DAYS_NAME, EXTERNAL_DRAWS_NAME)
        if not runs:
            raise SystemExit(f"{results_dir}: no output of external_validation.py (with {EXTERNAL_ALL_DAYS_NAME} and "
                             "model_dir in log.txt) in it or its subfolders. Check --results_dir.")  # fmt: skip
    else:
        runs = find_runs(results_dir)
        names = (ALL_DAYS_NAME, DRAWS_NAME)
        if not runs:
            raise SystemExit(f"{results_dir}: no run folders (…|target:<T>|train_frac:…|run:<r>). Check --results_dir.")

    tasks = label_tasks(sex, args.n_boot)
    rows, replicates = [], []
    for label, target, metric, definition, score in tasks:
        info = {"label": label, "metric": metric, "definition": definition, "target": target}
        target_runs = [(run, run_dir) for t, run, run_dir in runs if t == target]
        if not target_runs:
            rows.append({**info, "run": "—", "run_dir": "—", **NOT_SCORED})
        for run, run_dir in target_runs:
            print(f"{label}: run {run}", flush=True)
            path = predictions_path(run_dir, *names)
            result = dict(NOT_SCORED) if path is None else score(path)
            values = result.pop(REPLICATES_KEY, np.array([]))
            replicates.append(pd.DataFrame({"label": label, "metric": metric, "run": run, "run_dir": run_dir.name,
                                            "replicate": np.arange(1, len(values) + 1), "value": values}))  # fmt: skip
            rows.append({**info, "run": run, "run_dir": run_dir.name, "predictions": path and path.name, **result})

    known = {target for _, target, *_ in tasks}
    unknown = sorted({target for target, *_ in runs if target not in known})
    if unknown:
        print(f"⚠️ Not evaluated, unknown target: {', '.join(unknown)}")

    out_path = results_dir / OUTPUT_NAME
    pd.DataFrame(rows).to_csv(out_path, index=False)
    replicates_path = results_dir / REPLICATES_NAME
    if args.n_boot > 0:
        pd.concat(replicates, ignore_index=True).to_csv(replicates_path, index=False)

    overview = pd.DataFrame([overview_row(row) for row in rows])
    print(f"\nResults: {results_dir}\n")
    print(overview.to_string(index=False))
    print(f"\nvalue [interval]: mean over the day draws [2.5, 97.5 percentile over the day draws]; "
          f"bootstrap 95% CI: patient bootstrap, {args.n_boot} replicates, each the mean over its own day draws.")  # fmt: skip

    not_scored = overview.loc[overview["value [interval]"] == "—", "label"].tolist()
    if not_scored:
        print(f"\nWARNING: {len(not_scored)}/{len(overview)} rows not scored (no run / predictions, scaled units, "
              f"sex unknown, or no events): {', '.join(not_scored)}")  # fmt: skip
    print(f"\nWrote {out_path}" + (f"\nWrote {replicates_path}" if args.n_boot > 0 else ""))


def build_parser():
    p = argparse.ArgumentParser(description="C-index / AUROC of all STraTS Chronicle runs in one folder.")
    p.add_argument("--results_dir", type=str, default=None,
                   help="folder holding one subfolder per run (default ../outputs/chronicle, with --external "
                        "../outputs/chronicle_external)")  # fmt: skip
    p.add_argument("--cohort", type=str, required=True, help="cohort directory holding <name of the directory>.csv")
    p.add_argument("--external", action="store_true", help="evaluate the output of external_validation.py")
    p.add_argument("--n_boot", type=int, default=N_BOOTSTRAP, help="bootstrap replicates (0: no bootstrap)")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
