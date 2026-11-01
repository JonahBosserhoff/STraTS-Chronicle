import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


# metric per target type, as in chronicle.metrics: C-index (survival), Pearson r (regression),
# ROC AUC (binary); higher is better for all three
METRICS = {'survival':'c_index', 'regression':'correlation', 'binary':'auc'}


def score(target_type, pred, labels):
    """pred: [N] model output; labels: [N, 2] = (time, event) for survival, else [N]."""
    if target_type=='survival':
        # Harrell's C-index (chronicle.metrics.compute_c_index); lifelines expects a
        # predicted survival time, so the risk score is negated
        return concordance_index(labels[:,0], -pred, labels[:,1])
    if target_type=='regression':
        # Pearson r in float64 (chronicle.metrics.calc_correlations), on the scaled labels
        x, y = pred.astype(np.float64), labels.astype(np.float64)
        x, y = x-x.mean(), y-y.mean()
        denom = np.linalg.norm(x)*np.linalg.norm(y)
        return float((x*y).sum()/denom) if len(x)>=2 and denom>0 else np.nan
    # binary (chronicle.metrics.calc_auc_scores); AUC of the logits = AUC of the probabilities
    y = labels.astype(int)
    return roc_auc_score(y, pred) if len(np.unique(y))==2 else np.nan


class SurvivalEvaluator:
    """Metric of the Chronicle target (c_index / correlation / auc, see METRICS), computed as in
    Chronicle: one cut day per patient, mean ± std over the dataset's num_eval_draws fixed draws of
    cut days (Chronicle's C-index). all_days=True scores every cut day of every patient at once
    (std 0), which is how Chronicle scores regression and binary targets (every <CLS> token with
    a target).
    label_scaler: Chronicle's label scaler of a regression target, or None. The metric is computed
    on the scaled values; saved predictions (prediction, ground_truth) are mapped back to the
    original units with it, as Chronicle does (chronicle.evaluation._prediction_rows)."""
    def __init__(self, args, label_scaler=None):
        self.args = args
        self.metric = METRICS[args.target_type]
        self.label_scaler = label_scaler

    def evaluate(self, model, dataset, split, train_step, save_path=None, all_days=False):
        self.args.logger.write('\nEvaluating on split = '+split+(' (all days with a target)' if all_days else ''))
        if all_days:
            draws = [dataset.all_cuts(split)]
        else:
            draws = [(np.asarray(dataset.splits[split]), cut_days) for cut_days in dataset.eval_cuts[split]]
        bsz = self.args.eval_batch_size
        model.eval()
        scores, predictions = [], []
        for draw, (eval_ind, cut_days) in enumerate(draws):
            pred, labels = [], []
            for start in range(0, len(eval_ind), bsz):
                batch = dataset.get_batch(eval_ind[start:start+bsz], cut_days[start:start+bsz])
                labels.append(batch.pop('labels'))
                batch = {k:v.to(self.args.device) for k,v in batch.items()}
                with torch.no_grad():
                    pred.append(model(**batch).cpu())
            pred, labels = torch.cat(pred).numpy(), torch.cat(labels).numpy()
            scores.append(score(self.args.target_type, pred, labels))
            if save_path is not None:
                if self.args.target_type=='survival':
                    cols = {'prediction':pred, 'time':labels[:,0], 'event':labels[:,1]}
                elif self.label_scaler is not None: # original units, as Chronicle
                    unscale = lambda x:self.label_scaler.inverse_transform(
                        np.asarray(x, dtype=float).reshape(-1, 1)).ravel()
                    cols = {'prediction':unscale(pred), 'ground_truth':unscale(labels)}
                else:
                    cols = {'prediction':pred, 'ground_truth':labels}
                predictions.append(pd.DataFrame({'patient_id':dataset.ts_ids[eval_ind], 'draw':draw,
                                                 'day':cut_days, **cols}))
        # as Chronicle, a draw whose metric is undefined (one class, constant) is left out
        defined = [s for s in scores if not np.isnan(s)]
        if len(defined)<len(scores):
            self.args.logger.write('WARNING: '+self.metric+' undefined (one class / constant) in '
                                   +str(len(scores)-len(defined))+' of '+str(len(scores))+' draws, left out')
        result = {self.metric:np.mean(defined) if defined else np.nan,
                  self.metric+'_std':np.std(defined) if defined else np.nan}
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                                   +str(train_step)+': '+str(result))
        if save_path is not None:
            pd.concat(predictions).to_csv(save_path, index=False)
            self.args.logger.write('Saved predictions to '+save_path)
        return result
