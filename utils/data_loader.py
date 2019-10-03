# -*- coding: utf-8 -*-

import codecs

from sklearn.model_selection import train_test_split

from .trie import Trie
from .text import TextSample, normalized_word, get_bigrams
from .io import pickle_load, format_filename
from config import PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, TEST_DATA_TEMPLATE


def load_ner_data(filename, normalized=False, lower=False, delimiter='\t', split_mode=0,
                  split_size=0.1, seed=42):
    """Load ner data from a given file.

    The file should follow CoNLL format:
    Each line is a token and its label separated by 'delimiter', or a blank line indicating the end
    of a sentence.

    Args:
        normalized: boolean, whether to normalized token
        lower: boolean
        filename: str, path to ner file
        delimiter: str, delimiter to split token and label
        split_mode: int, if `split_mode` is 1, it will split the dataset into train and valid;
                         if `split_mode` is 2, it will split the dataset into train, valid and test.
        split_size: float, the proportion of test subset, between 0.0 and 1.0
        seed: int, random seed

    """
    print(f'Logging Info - Reading ner file: {filename}')
    with codecs.open(filename, 'r', encoding='utf8') as reader:
        data_list = []

        # tokens, labels = [], []
        text_sample = TextSample()
        for i, line in enumerate(reader):
            line = line.rstrip()
            if line:
                line_split = line.split(delimiter)
                if len(line_split) == 2:
                    token, tag = line_split
                    if lower:
                        token = token.lower()
                    if normalized:
                        token = normalized_word(token)
                    text_sample.tokens.append(token)
                    text_sample.tags.append(tag)
                else:
                    raise Exception(
                        f'Format Error at line {i}! Input file should follow CoNLL format.'
                    )
            else:
                if text_sample.tokens:
                    text_sample.fw_bigrams, text_sample.bw_bigrams = get_bigrams(text_sample.tokens)
                    text_sample.raw_text = ''.join(text_sample.tokens)
                    data_list.append(text_sample)
                    text_sample = TextSample()

        if text_sample.tokens:  # in case there's no blank line at the end of the file
            text_sample.fw_bigrams, text_sample.bw_bigrams = get_bigrams(text_sample.tokens)
            text_sample.raw_text = ''.join(text_sample.tokens)
            data_list.append(text_sample)

    if split_mode == 1:
        train_data, test_data = train_test_split(data_list, test_size=split_size, random_state=seed)
        return train_data, test_data
    elif split_mode == 2:
        train_data, holdout_data = train_test_split(data_list, test_size=split_size*2,
                                                    random_state=seed)
        dev_data, test_data = train_test_split(holdout_data, test_size=0.5, random_state=seed)
        return train_data, dev_data, test_data
    else:
        return data_list


def load_gaze_trie(filename, normalized=False, lower=False, delimiter=' '):
    print(f'Logging Info - Reading gaze file: {filename}')
    trie = Trie()
    with codecs.open(filename, 'r', encoding='utf8')as reader:
        for line in reader:
            line = line.strip()
            if line == '':
                continue
            word = line.split(delimiter)[0]
            if lower:
                word = word.lower()
            if normalized:
                word = normalized_word(word)
            trie.insert(word)
    return trie


def load_processed_data(data_type, dataset):
    if data_type == 'train':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE,
                                           dataset=dataset))
    elif data_type == 'dev':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, dataset=dataset))
    elif data_type == 'test':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, dataset=dataset))
    else:
        raise ValueError('data tye not understood: {}'.format(data_type))
    return data
