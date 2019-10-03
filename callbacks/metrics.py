# -*- coding: utf-8 -*-

from keras.callbacks import Callback

from utils import ner_eval


class NERMetric(Callback):
    """
    callback for evaluating ner model
    """
    def __init__(self, model_input, gold_tags, idx2tag):
        self.idx2tag = idx2tag
        self.model_input = model_input
        self.gold_tags = gold_tags
        super(NERMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        pred_probs = self.model.predict(self.model_input)
        r, p, f1 = ner_eval(self.gold_tags, self.idx2tag, pred_probs)

        logs['val_r'] = r
        logs['val_p'] = p
        logs['val_f1'] = f1
        print('Epoch {}: val_r: {}, val_p: {}, val_f1: {}'.format(epoch+1, r, p, f1))
