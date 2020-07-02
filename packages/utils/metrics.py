import pandas as pd
import numpy as np
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from sklearn.metrics import accuracy_score


class Predict(Metric):

    def __init__(self, cfg):
        self.df = pd.DataFrame(columns=['filename', 'category'])
        self.prediction_csv_path = cfg.PREDICTION_CSV
        super(Predict, self).__init__()

    @reinit__is_reduced
    def reset(self):
        self.df = pd.DataFrame(columns=['filename', 'category'])
        super(Predict, self).reset()

    @reinit__is_reduced
    def update(self, batch_outputs):
        preds, _, filenames = batch_outputs
        self.df = self.df.append(pd.DataFrame(list(zip(filenames, preds)), columns=self.df.columns))

    @sync_all_reduce()
    def compute(self):
        tmp = self.df['category'].tolist()
        self.df['category'] = [f{cls:02} for cls in tmp]
        self.df.to_csv(self.prediction_csv_path, index=False)


class TOP_1(Metric):

    def __init__(self):
        self.preds = np.array([])
        self.labels = np.array([])
        super(TOP_1, self).__init__()

    @reinit__is_reduced
    def reset(self):
        self.preds = np.array([])
        self.labels = np.array([])
        super(TOP_1, self).reset()

    @reinit__is_reduced
    def update(self, batch_outputs):
        preds, labels, _ = batch_outputs
        self.preds = np.append(self.preds, preds)
        self.labels = np.append(self.labels, labels)

    @sync_all_reduce()
    def compute(self):
        top_1 = accuracy_score(self.labels, self.preds)
        return top_1
