# -*- coding: utf-8 -*

import numpy as np
from seqeval import metrics


def ner_tag_decode(idx2tag, pred_probs, lengths=None):
    pred_ids = np.argmax(pred_probs, axis=-1)
    pred_tags = [[idx2tag[tag_id] for tag_id in ids] for ids in pred_ids]
    if lengths is not None:
        pred_tags = [tags[:length] for tags, length in zip(pred_tags, lengths)]
    return pred_tags


def ner_eval(gold_tags, idx2tag, pred_probs):
    lengths = [min(len(tag), pred_prob.shape[0])
               for tag, pred_prob in zip(gold_tags, pred_probs)]
    pred_tags = ner_tag_decode(idx2tag, pred_probs, lengths)

    r = metrics.recall_score(gold_tags, pred_tags)
    p = metrics.precision_score(gold_tags, pred_tags)
    f1 = metrics.f1_score(gold_tags, pred_tags)
    return r, p, f1
