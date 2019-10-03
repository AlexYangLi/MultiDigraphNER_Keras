# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.models import Model

from .base_model import BaseModel
from layers import GGNN
from callbacks import NERMetric
from utils import ner_tag_decode, ner_eval


class MultiDigraph(BaseModel):
    def __init__(self, config):
        super(MultiDigraph, self).__init__(config)

    def add_metrics(self, dev_generator):
        dev_input, _ = dev_generator.prepare_input(batch_indices=range(dev_generator.data_size))
        dev_tags = [text_example.tags for text_example in dev_generator.data]
        self.callbacks.append(NERMetric(dev_input, dev_tags, self.config.idx2tag))
        print('Logging Info - Callback Added: NERMetrics...')

    def build(self):
        input_char = Input(shape=(None, ), name='input_char')
        input_fw_bigram = Input(shape=(None, ), name='input_fw_bigram')
        input_bw_bigram = Input(shape=(None, ), name='input_bw_bigram')
        input_adj_matrix = Input(shape=(None, None, None),
                                 name='input_adj_matrix')
        input_gaze = Input(shape=(None, ), name='input_gaze')
        inputs = [input_char, input_fw_bigram, input_bw_bigram, input_adj_matrix, input_gaze]

        char_embeddings = Embedding(input_dim=self.config.char_embedding.shape[0],
                                    output_dim=self.config.char_embedding.shape[1],
                                    mask_zero=True,
                                    weights=[self.config.char_embedding],
                                    trainable=self.config.char_trainable,
                                    name='char_embeddings')
        fw_bigram_embeddings = Embedding(input_dim=self.config.fw_bigram_embeddings.shape[0],
                                         output_dim=self.config.fw_bigram_embeddings.shape[1],
                                         mask_zero=True,
                                         weights=[self.config.bw_bigram_embeddings],
                                         trainable=self.config.fw_bigram_trainable,
                                         name='fw_bigram_embeddings')
        bw_bigram_embeddings = Embedding(input_dim=self.config.bw_bigram_embeddings.shape[0],
                                         output_dim=self.config.bw_bigram_embeddings.shape[1],
                                         mask_zero=True,
                                         weights=[self.config.bw_bigram_embeddings],
                                         trainable=self.config.bw_bigram_trainable,
                                         name='bw_bigram_embeddings')
        gaze_embeddings = Embedding(input_dim=self.config.n_gaze * 2,
                                    output_dim=self.config.gaze_embed_dim,
                                    mask_zero=False,
                                    trainable=True,
                                    name='gaze_embeddings')

        char_embed = char_embeddings(input_char)
        fw_bigram_embed = fw_bigram_embeddings(input_fw_bigram)
        bw_bigram_embed = bw_bigram_embeddings(input_bw_bigram)
        token_embed = concatenate([char_embed, fw_bigram_embed, bw_bigram_embed], axis=-1)
        token_embed = SpatialDropout1D(self.config.dropout)(token_embed)
        token_state = TimeDistributed(Dense(self.config.graph_embed_dim))(token_embed)

        gaze_embed = gaze_embeddings(input_gaze)
        gaze_state = Dense(self.config.graph_embed_dim)(gaze_embed)  # TODO: use multi linear layer

        init_state = Lambda(lambda x: self.concat_token_gaze(x[0], x[1], x[2]))(
            [input_char, token_state, gaze_state]
        )

        graph_embed_list = [init_state]
        for i in range(self.config.n_layer):
            graph_embeddings = GGNN(units=self.config.graph_embed_dim,
                                    n_gaze=self.config.n_gaze,
                                    n_step=self.config.n_step,
                                    name=f'ggnn_{i+1}')
            graph_embed = graph_embeddings([graph_embed_list[-1], input_adj_matrix])
            graph_embed_list.append(graph_embed)

        token_node_embed = Lambda(lambda x: self.get_token_node(x[0], x[1]))(
            [input_char, graph_embed_list[-1]]
        )
        token_node_embed = Masking()(token_node_embed)

        lstm_embed_1 = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))(
            token_node_embed)
        lstm_embed_2 = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))(
            lstm_embed_1)
        input_encode = TimeDistributed(Dense(len(self.config.tag_vocab)))(lstm_embed_2)
        input_encode = Lambda(lambda x: tf.nn.log_softmax(x))(input_encode)

        ner_tag = CRF(units=len(self.config.tag_vocab))(input_encode)
        ner_model = Model(inputs, ner_tag)
        ner_model.compile(optimizer=self.config.optimizer, loss=crf_loss, metrics=[crf_accuracy])
        return ner_model

    def concat_token_gaze(self, input_token, token_state, gaze_state):
        batch_size = K.shape(token_state)[0]
        max_len = K.shape(token_state)[1]
        embed_dim = K.shape(token_state)[2]

        # [batch_size, 1]
        seq_length = K.sum(K.cast(K.equal(input_token, 0), 'int32'), axis=-1, keepdims=True)

        # [batch_size, n_gaze*2]
        gaze_index = K.tile(K.expand_dims(K.arange(self.config.n_gaze * 2), axis=0),
                            [batch_size, 1])
        gaze_index += seq_length
        # [batch_size, n_gaze*2]
        batch_index = K.tile(K.expand_dims(K.arange(batch_size), 1), [1, self.config.n_gaze * 2])
        # [batch_size*n_gaze*2, 2]
        gaze_index = K.reshape(K.stack([gaze_index, batch_index], axis=2), (-1, 2))

        gaze_scatter = tf.scatter_nd(indices=gaze_index,
                                     updates=K.reshape(gaze_state, (-1, embed_dim)),
                                     shape=(batch_size, max_len, embed_dim))
        return token_state + gaze_scatter

    def get_token_node(self, input_token, graph_embed):
        seq_mask = K.expand_dims(K.cast(K.equal(input_token, 0), K.floatx()), axis=-1)
        mask_graph_embed = graph_embed * seq_mask
        return mask_graph_embed[:, :-self.config.n_gaze * 2, :]

    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    def fit_generator(self, train_generator, dev_generator):
        self.callbacks = []
        self.add_metrics(dev_generator)
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit_generator(generator=train_generator,
                                 epochs=self.config.n_epoch,
                                 callbacks=self.callbacks,
                                 validation_data=dev_generator)
        print('Logging Info - Training end...')

    def predict(self, model_input, lengths=None):
        pred_probs = self.model.predict(model_input)
        return ner_tag_decode(self.config.idx2tag, pred_probs, lengths=None)

    def evaluate(self, model_input, gold_tags):
        pred_probs = self.model.predict(model_input)
        r, p, f1 = ner_eval(gold_tags, self.config.idx2tag, pred_probs)
        print(f'Logging Info - recall: {r}, precision: {p}, f1: {f1}')
        return r, p, f1
