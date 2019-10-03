# -*- coding: utf-8 -*-

import os
import time
import gc

import numpy as np
from keras.optimizers import *

from config import ModelConfig, EMBEDDING_MATRIX_TEMPLATE, VOCABULARY_TEMPLATE, \
    PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, LOG_DIR, PERFORMANCE_LOG_TEMPLATE
from utils import format_filename, pickle_load, NERGenerator, write_log
from models import MultiDigraph


def train(dataset, char_embed_type, char_trainable, fw_embed_type, fw_trainable, bw_embed_type,
          bw_trainable, gaze_embed_dim, n_step, n_layer, rnn_units, dropout, batch_size, n_epoch,
          optimizer, callbacks_to_add=None, overwrite=False):
    config = ModelConfig()
    config.char_embedding = np.load(format_filename(PROCESSED_DATA_DIR,
                                                    EMBEDDING_MATRIX_TEMPLATE,
                                                    dataset=dataset,
                                                    type=char_embed_type))
    config.char_trainable = char_trainable
    config.fw_bigram_embeddings = np.load(format_filename(PROCESSED_DATA_DIR,
                                                          EMBEDDING_MATRIX_TEMPLATE,
                                                          dataset=dataset,
                                                          type=fw_embed_type))
    config.fw_bigram_trainable = fw_trainable
    config.bw_bigram_embeddings = np.load(format_filename(PROCESSED_DATA_DIR,
                                                          EMBEDDING_MATRIX_TEMPLATE,
                                                          dataset=dataset,
                                                          type=bw_embed_type))
    config.bw_bigram_trainable = bw_trainable
    config.char_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE,
                                                    dataset=dataset, level='char'))
    config.fw_bigram_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE,
                                                         dataset=dataset, level='fw_bigram'))
    config.bw_bigram_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE,
                                                         dataset=dataset, level='bw_bigram'))
    config.tag_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE,
                                                   dataset=dataset, level='tag'))
    config.idx2tag = pickle_load(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE,
                                                 dataset=dataset, level='tag'))

    config.gaze_embed_dim = gaze_embed_dim
    config.n_step = n_step
    config.n_layer = n_layer
    config.rnn_units = rnn_units
    config.dropout = dropout
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.optimizer = optimizer
    config.callbacks_to_add = callbacks_to_add

    config.model_name = 'ggnn'
    config.exp_name = 'ggnn_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        char_embed_type, 'tune' if char_trainable else 'fix',
        fw_embed_type, 'tune' if fw_trainable else 'fix',
        bw_embed_type, 'tune' if bw_embed_type else 'fix',
        gaze_embed_dim, n_step, n_layer, rnn_units, batch_size, n_epoch
    )
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    dev_generator = NERGenerator('dev', dataset, batch_size, config.char_vocab,
                                 config.fw_bigram_vocab, config.bw_bigram_vocab, config.tag_vocab)
    dev_input, _ = dev_generator.prepare_input(range(dev_generator.data_size))
    dev_tags = [text_example.tags for text_example in dev_generator.data]

    test_generator = NERGenerator('test', dataset, batch_size, config.char_vocab,
                                  config.fw_bigram_vocab, config.bw_bigram_vocab, config.tag_vocab)
    test_input, _ = test_generator.prepare_input(range(test_generator.data_size))
    test_tags = [text_example.tags for text_example in test_generator.data]

    config.n_gaze = dev_generator.n_gaze

    # logger to log output of training process
    train_log = {'exp_name': config.exp_name}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = MultiDigraph(config)

    if not os.path.exists(model_save_path) or overwrite:
        train_generator = NERGenerator('train', dataset, batch_size, config.char_vocab,
                                       config.fw_bigram_vocab, config.bw_bigram_vocab,
                                       config.tag_vocab)
        start_time = time.time()
        model.fit_generator(train_generator, dev_generator)
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    model.load_best_model()
    print('Logging Info - Evaluate over valid data:')
    dev_score = model.evaluate(dev_input, dev_tags)
    train_log['dev_performance'] = dev_score
    print('Logging Info - Evaluate over test data:')
    test_score = model.evaluate(test_input, test_tags)
    train_log['test_performance'] = test_score

    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over valid data based on swa model:')
        swa_dev_score = model.evaluate(dev_input, dev_tags)
        train_log['swa_dev_performance'] = swa_dev_score
        print('Logging Info - Evaluate over test data based on swa model:')
        swa_test_score = model.evaluate(test_input, test_tags)
        train_log['swa_test_performance'] = swa_test_score

    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG_TEMPLATE, dataset=dataset),
              log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()


if __name__ == '__main__':
    train(dataset='ecommerce',
          char_embed_type='c2v',
          char_trainable=True,
          fw_embed_type='fw_bi2v',
          fw_trainable=True,
          bw_embed_type='bw_bi2v',
          bw_trainable=True,
          gaze_embed_dim=50,
          n_step=2,
          n_layer=1,
          rnn_units=600,
          dropout=0.5,
          batch_size=10,
          n_epoch=100,
          optimizer=SGD(lr=0.01, momentum=0, decay=0.05),
          callbacks_to_add=['modelcheckpoint'])
    train(dataset='ecommerce',
          char_embed_type='c2v',
          char_trainable=True,
          fw_embed_type='fw_bi2v',
          fw_trainable=True,
          bw_embed_type='bw_bi2v',
          bw_trainable=True,
          gaze_embed_dim=50,
          n_step=2,
          n_layer=1,
          rnn_units=600,
          dropout=0.5,
          batch_size=10,
          n_epoch=100,
          optimizer=SGD(lr=0.01, momentum=0, decay=0.05),
          callbacks_to_add=['modelcheckpoint', 'earlystopping', 'swa'])
