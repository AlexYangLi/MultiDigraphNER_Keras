# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = './raw_data'
PROCESSED_DATA_DIR = './data'
LOG_DIR = './log'
MODEL_SAVED_DIR = './ckpt'

PAD = '<PAD>'
UNK = '<UNK>'
CLS = '<CLS>'
SEQ = '<SEQ>'

NER_TRAIN_FILE = {
    'ecommerce': os.path.join(RAW_DATA_DIR, 'ecommerce/train.txt.bieos')
}
NER_DEV_FILE = {
    'ecommerce': os.path.join(RAW_DATA_DIR, 'ecommerce/dev.txt.bieos')
}
NER_TEST_FILE = {
    'ecommerce': os.path.join(RAW_DATA_DIR, 'ecommerce/test.txt.bieos')
}


def list_dir(path):
    return [os.path.join(path, file) for file in os.listdir(path)]


GAZETTEER_FILES = {
    'ecommerce': list_dir(os.path.join(RAW_DATA_DIR, 'dics/ecommerce'))
}

EXTERNAL_EMBEDDING_DIR = os.path.join(RAW_DATA_DIR, 'embeddings')
EXTERNAL_EMBEDDING_FILE = {
    'char': os.path.join(EXTERNAL_EMBEDDING_DIR, 'char.vec200.demo'),
    'bigram': os.path.join(EXTERNAL_EMBEDDING_DIR, 'bichar.vec200.demo')
}

VOCABULARY_TEMPLATE = '{dataset}_{level}_vocab.pkl'
IDX2TOKEN_TEMPLATE = '{dataset}_idx2{level}.pkl'
EMBEDDING_MATRIX_TEMPLATE = '{dataset}_{type}_embeddings.npy'

TRAIN_DATA_TEMPLATE = '{dataset}_train.pkl'
DEV_DATA_TEMPLATE = '{dataset}_dev.pkl'
TEST_DATA_TEMPLATE = '{dataset}_test.pkl'

PERFORMANCE_LOG_TEMPLATE = '{dataset}_performance.log'


class ProcessConfig(object):
    """Configuration for pre-processing data"""
    def __init__(self):
        self.normalized = False
        self.lower = True
        self.char_embed_dim = 200
        self.bigram_embed_dim = 200


class ModelConfig:
    """Configuration for model"""
    def __init__(self):
        self.char_embedding = None
        self.char_trainable = True
        self.fw_bigram_embeddings = None
        self.fw_bigram_trainable = True
        self.bw_bigram_embeddings = None
        self.bw_bigram_trainable = True
        self.char_vocab = None
        self.fw_bigram_vocab = None
        self.bw_bigram_vocab = None
        self.tag_vocab = None
        self.idx2tag = None

        self.n_gaze = None
        self.gaze_embed_dim = 50
        self.graph_embed_dim = 300
        self.n_step = 2
        self.n_layer = 1

        self.rnn_units = 600
        self.dropout = 0.5

        self.exp_name = None
        self.model_name = None

        self.batch_size = 10
        self.n_epoch = 100
        self.optimizer = None
        self.optimizer_type = 'sgd'
        self.lr = 0.001
        self.lr_decay = 0.05
        self.momentum = 0
        self.l2_weight = 1e-8
        self.clip = 5

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_f1'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stopping configuration
        self.early_stopping_monitor = 'val_f1'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 3
        self.early_stopping_verbose = 1

        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 5
