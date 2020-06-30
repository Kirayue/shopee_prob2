import pandas as pd
import numpy as np
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Predict(Metric):

    def __init__(self, cfg, output_transform=lambda x: x):
        self.df = pd.DataFrame(columns=['filename', 'category'])
        self.prediction_csv_path = cfg.PREDICTION_CSV
        super(Predict, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.df = pd.DataFrame(columns=['filename', 'category'])
        super(Predict, self).reset()

    @reinit__is_reduced
    def update(self, outputs):
        output, labels, infoes = outputs
        out = []
        for prob, info in zip(output, infoes):
            LNDbID = np.array(info["LNDbID"]).reshape(1, 1)
            center = info["center"].reshape(1, 3)
            r = info["r"].reshape(1, 3)
            detection_prob = info["prob"].reshape(1, 1)
            out.append(np.concatenate((LNDbID, center, prob.reshape(1, 1), r, detection_prob), 1))
        out = np.concatenate(out, 0)
        self.df = self.df.append(pd.DataFrame(out, columns=self.df.columns))

    @sync_all_reduce()
    def compute(self):
        self.df = self.df.astype({'seriesuid': 'int32'})
        self.df.to_csv(self.prediction_csv_path, index=False)


class TOP_1(Predict):

    def __init__(self, cfg):
        super(TOP_1, self).__init__(cfg)
        self.annotation_csv_path = cfg.GT_CSV
        self.annotation_excluded_csv_path = cfg.EXCLUDED_CSV
        self.seriesuids_csv_path = cfg.SERIESUIDS_CSV
        self.eval_dir_path = cfg.OUTPUT_DIR

    @sync_all_reduce()
    def compute(self):
        super(FROC, self).compute()

        results = noduleCADEvaluation(
            self.annotation_csv_path,
            self.annotation_excluded_csv_path,
            self.seriesuids_csv_path,
            self.prediction_nms_csv_path,
            self.eval_dir_path
        )
        return results[-1]
