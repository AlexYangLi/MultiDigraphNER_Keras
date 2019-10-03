# -*- coding: utf-8 -*-

import re


def normalized_word(token):
    return re.sub(r'[0-9]', '0', token)


def get_bigrams(tokens):
    fw_bigrams, bw_bigrams = [], []
    for j in range(len(tokens)):
        if j == len(tokens) - 1:
            fw_bigrams.append('<END>')
        else:
            fw_bigrams.append(tokens[j] + tokens[j + 1])

        if j == 0:
            bw_bigrams.append('<START>')
        else:
            bw_bigrams.append(tokens[j] + tokens[j - 1])
    return fw_bigrams, bw_bigrams


class TextSample(object):
    def __init__(self):
        self.raw_text = ''
        self.tokens = []
        self.fw_bigrams = []
        self.bw_bigrams = []
        self.tags = []
        self.gaze_match = dict()
