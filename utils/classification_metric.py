import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import math


class Classification(object):
    def __init__(self):
        self.init()

    def init(self):
        self.pred_list = []
        self.label_list = []
        self.correct_count = 0
        self.total_count = 0
        self.loss = 0
        self.proba = []
        self.converted_proba = []

    def update(self, pred, label, easy_model=False):
        pred = pred.cpu()
        self.proba.extend(pred)

        self.converted_proba = [[math.exp(i) / sum(math.exp(i) for i in self.proba[index]) for i in self.proba[index]]
                                for index in range(len(self.proba))]
        label = label.cpu()

        if easy_model:
            pass
        else:
            loss = F.cross_entropy(pred, label).item() * len(label)
            self.loss += loss
            pred = pred.data.max(1)[1]
        self.pred_list.extend(pred.numpy())
        self.label_list.extend(label.numpy())
        self.correct_count += pred.eq(label.data.view_as(pred)).sum()
        self.total_count += len(label)

    def results(self):
        result_dict = {}
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        result_dict['auc'] = float(roc_auc_score(self.label_list, self.converted_proba, multi_class="ovr"))
        self.init()
        return result_dict
