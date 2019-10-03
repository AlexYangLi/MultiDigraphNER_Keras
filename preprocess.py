# -*- coding: utf-8 -*-

import os
from collections import Counter

import numpy as np

from config import NER_TRAIN_FILE, NER_DEV_FILE, NER_TEST_FILE, GAZETTEER_FILES, \
    TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, TEST_DATA_TEMPLATE, PROCESSED_DATA_DIR, \
    VOCABULARY_TEMPLATE, IDX2TOKEN_TEMPLATE, EMBEDDING_MATRIX_TEMPLATE, PAD, UNK, CLS, SEQ, \
    ProcessConfig, MODEL_SAVED_DIR, LOG_DIR
from utils import load_ner_data, load_gaze_trie, train_w2v, train_fasttext, train_glove, \
    pickle_dump, format_filename


def search_entity(text_examples, gaze_trie):
    for gaze_name in gaze_trie:
        for text_example in text_examples:
            text_example.gaze_match[gaze_name] = gaze_trie[gaze_name].match(text_example.tokens)


def build_vocab(corpus, min_count=1, special_token='standard'):
    token_count = Counter()
    for text in corpus:
        token_count.update(text)

    if special_token == 'standard':
        token_vocab = {PAD: 0, UNK: 1}
    elif special_token == 'bert':
        token_vocab = {PAD: 0, UNK: 1, CLS: 2, SEQ: 3}
    else:
        raise ValueError('Argument `special_token` can only be "standard" or "bert", '
                         'got: {}'.format(special_token))
    for token, count in token_count.items():
        if count >= min_count:
            token_vocab[token] = len(token_vocab)
    idx2token = dict((idx, token) for token, idx in token_vocab.items())
    print(f'Logging Info - Build vocabulary finished, vocabulary size: {len(token_vocab)}')
    return token_vocab, idx2token


def build_tag_vocab(labels):
    """Build label vocabulary

    Args:
        labels: list of list of str, the label strings
    """
    tag_count = Counter()
    for sequence in labels:
        tag_count.update(sequence)

    # sorted by frequency, so that the label with the highest frequency will be given
    # id of 0, which is the default id for unknown labels
    sorted_tag_count = dict(tag_count.most_common())

    tag_vocab = {}
    for tag in sorted_tag_count:
        tag_vocab[tag] = len(tag_vocab)

    id2tag = dict((idx, tag) for tag, idx in tag_vocab.items())

    print(f'Build label vocabulary finished, vocabulary size: {len(tag_vocab)}')
    return tag_vocab, id2tag


def process_data(dataset: str, config: ProcessConfig):
    train_file = NER_TRAIN_FILE[dataset]
    dev_file = NER_DEV_FILE.get(dataset, None)
    test_file = NER_TEST_FILE.get(dataset, None)

    print('Logging Info - Loading ner data...')
    if dev_file is None and test_file is None:
        train_data, dev_data, test_data = load_ner_data(train_file, config.normalized, config.lower,
                                                        split_mode=2)
    elif dev_file is None:
        train_data, dev_data = load_ner_data(train_file, config.normalized, config.lower,
                                             split_mode=1)
        test_data = load_ner_data(test_file, config.normalized, config.lower)
    elif test_file is None:
        train_data, test_data = load_ner_data(train_file, config.normalized, config.lower,
                                              split_mode=1)
        dev_data = load_ner_data(dev_file, config.normalized)
    else:
        train_data = load_ner_data(train_file, config.normalized, config.lower)
        dev_data = load_ner_data(dev_file, config.normalized, config.lower)
        test_data = load_ner_data(test_file, config.normalized, config.lower)

    print('Logging Info - Loading gazetteer and generating trie...')
    gaze_tries = dict()
    for gaze_file in GAZETTEER_FILES[dataset]:
        gaze_name = os.path.basename(gaze_file)
        gaze_tries[gaze_name] = load_gaze_trie(gaze_file, config.normalized, config.lower)

    print('Logging Info - Generating matching entity...')
    search_entity(train_data, gaze_tries)
    search_entity(dev_data, gaze_tries)
    search_entity(test_data, gaze_tries)

    print('Logging Info - Generating corpus...')
    char_corpus = [text_example.tokens for text_example in train_data+dev_data+test_data]
    fw_bigram_corpus = [text_example.fw_bigrams for text_example in train_data+dev_data+test_data]
    bw_bigram_corpus = [text_example.bw_bigrams for text_example in train_data+dev_data+test_data]
    tag_corpus = [text_example.tags for text_example in train_data+dev_data+test_data]

    print('Logging Info - Generating vocabulary...')
    char_vocab, idx2char = build_vocab(char_corpus)
    fw_bigram_vocab, idx2fw_bigram = build_vocab(fw_bigram_corpus)
    bw_bigram_vocab, idx2bw_bigram = build_vocab(bw_bigram_corpus)
    tag_vocab, idx2tag = build_tag_vocab(tag_corpus)

    print('Logging Info - Preparing embedding...')
    c2v = train_w2v(char_corpus, char_vocab, embedding_dim=config.char_embed_dim)
    c_fasttext = train_fasttext(char_corpus, char_vocab, embedding_dim=config.char_embed_dim)
    c_glove = train_glove(char_corpus, char_vocab, embedding_dim=config.char_embed_dim)
    fw_bi2v = train_w2v(fw_bigram_corpus, fw_bigram_vocab, embedding_dim=config.bigram_embed_dim)
    fw_bifasttext = train_fasttext(fw_bigram_corpus, fw_bigram_vocab,
                                   embedding_dim=config.bigram_embed_dim)
    fw_biglove = train_glove(fw_bigram_corpus, fw_bigram_vocab,
                             embedding_dim=config.bigram_embed_dim)
    bw_bi2v = train_w2v(bw_bigram_corpus, bw_bigram_vocab, embedding_dim=config.bigram_embed_dim)
    bw_bifasttext = train_fasttext(bw_bigram_corpus, bw_bigram_vocab,
                                   embedding_dim=config.bigram_embed_dim)
    bw_biglove = train_glove(bw_bigram_corpus, bw_bigram_vocab,
                             embedding_dim=config.bigram_embed_dim)

    print('Logging Info - Saving processed data...')
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, dataset=dataset),
                train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, dataset=dataset),
                dev_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, dataset=dataset),
                test_data)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, dataset=dataset,
                                level='char'),
                char_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, dataset=dataset,
                                level='fw_bigram'),
                fw_bigram_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, dataset=dataset,
                                level='bw_bigram'),
                bw_bigram_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, dataset=dataset,
                                level='tag'),
                tag_vocab)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, dataset=dataset,
                                level='char'),
                idx2char)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, dataset=dataset,
                                level='fw_bigram'),
                idx2fw_bigram)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, dataset=dataset,
                                level='bw_bigram'),
                idx2fw_bigram)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, dataset=dataset,
                                level='tag'),
                idx2tag)

    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='c2v'),
            c2v)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='c_fasttext'),
            c_fasttext)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='c_glove'),
            c_glove)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='fw_bi2v'),
            fw_bi2v)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='fw_bifasttext'),
            fw_bifasttext)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='fw_biglove'),
            fw_biglove)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='bw_bi2v'),
            bw_bi2v)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='bw_bifasttext'),
            bw_bifasttext)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, dataset=dataset,
                            type='bw_biglove'),
            bw_biglove)


if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)

    process_data('ecommerce', config=ProcessConfig())
