import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index


class SurvivalEvaluator:
    """Harrell's C-index of the OS risk score, computed as in Chronicle (chronicle.metrics.compute_c_index):
    one cut day per patient, mean ± std over the dataset's num_eval_draws fixed draws of cut days."""
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step, save_path=None):
        self.args.logger.write('\nEvaluating on split = '+split)
        eval_ind = dataset.splits[split]
        bsz = self.args.eval_batch_size
        model.eval()
        c_indices, predictions = [], []
        for draw, cut_days in enumerate(dataset.eval_cuts[split]):
            risk, labels = [], []
            for start in range(0, len(eval_ind), bsz):
                batch = dataset.get_batch(eval_ind[start:start+bsz], cut_days[start:start+bsz])
                labels.append(batch.pop('labels'))
                batch = {k:v.to(self.args.device) for k,v in batch.items()}
                with torch.no_grad():
                    risk.append(model(**batch).cpu())
            risk, labels = torch.cat(risk).numpy(), torch.cat(labels).numpy()
            # lifelines expects a predicted survival time, so the risk score is negated
            c_indices.append(concordance_index(labels[:,0], -risk, labels[:,1]))
            if save_path is not None:
                predictions.append(pd.DataFrame({'patient_id':dataset.ts_ids[eval_ind], 'draw':draw,
                                                 'day':cut_days, 'prediction':risk,
                                                 'time':labels[:,0], 'event':labels[:,1]}))
        result = {'c_index':np.mean(c_indices), 'c_index_std':np.std(c_indices)}
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                                   +str(train_step)+': '+str(result))
        if save_path is not None:
            pd.concat(predictions).to_csv(save_path, index=False)
            self.args.logger.write('Saved predictions to '+save_path)
        return result
