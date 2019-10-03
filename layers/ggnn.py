# -*- coding: utf-8 -*-

import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import GRUCell
from keras.initializers import RandomNormal


class GGNN(Layer):
    """
    Implementation of Adapted GGNN introduced in Ding et.al.
    "A Neural Multi-digraph Model for Chinese NER with Gazetteers"
    """
    def __init__(self, units, n_gaze, n_step, **kwargs):
        super(GGNN, self).__init__(**kwargs)
        self.units = units
        self.n_gaze = n_gaze
        self.n_edge = (self.n_gaze + 1) * 2
        self.n_step = n_step
        self.gru_cell = GRUCell(units=self.units)

    def build(self, input_shape):
        embed_dim = input_shape[0][-1]
        assert embed_dim == self.units
        self.alpha = self.add_weight(name=self.name+'contribution_coefficient',
                                     shape=(self.n_edge, ),
                                     initializer='ones')
        self.w = self.add_weight(name=self.name+'_w',
                                 shape=(self.n_edge, embed_dim, self.units),
                                 initializer=RandomNormal(0., 0.02))
        self.b = self.add_weight(name=self.name+'_b',
                                 shape=(self.n_edge, self.units),
                                 initializer='zeros')
        self.gru_cell.build([None, self.units * self.n_edge])
        super(GGNN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # init_state: [batch_size, n_node, embed_dim]
        # adj_matrix: [batch_size, n_edge, n_node, n_node]
        init_state, adj_matrix = inputs
        n_node = K.shape(init_state)[1]

        expand_alpha = K.expand_dims(K.expand_dims(self.alpha, axis=-1), axis=-1)
        weighted_adj_matrix = adj_matrix * K.sigmoid(expand_alpha)

        cur_state = K.identity(init_state)
        for _ in range(self.n_step):
            h = K.dot(cur_state, self.w) + self.b  # [batch_size, n_node, n_edge, units]
            neigh_state = []
            for edge_idx in range(self.n_edge):
                neigh_state.append(K.batch_dot(weighted_adj_matrix[:, edge_idx, :, :],
                                               h[:, :, edge_idx, :],
                                               axes=(2, 1)))    # [batch_size, n_node, units]
            neigh_state = K.concatenate(neigh_state, axis=-1)   # [batch_size, n_node, units*n_edge]

            gru_inputs = K.reshape(neigh_state, (-1, self.units * self.n_edge))
            gru_states = K.reshape(cur_state, (-1, self.units))
            # should look up into GRUCell's implementation
            gru_output, _ = self.gru_cell.call(inputs=gru_inputs,
                                               states=[gru_states])
            cur_state = K.reshape(gru_output, (-1, n_node, self.units))
        return cur_state

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.units

    @property
    def trainable_weights(self):
        return self._trainable_weights + self.gru_cell.trainable_weights
