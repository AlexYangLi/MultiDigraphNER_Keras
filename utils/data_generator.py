# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import Sequence, to_categorical

from utils import load_processed_data, pad_sequences_1d, \
    pad_sequences_2d
from config import UNK


class NERGenerator(Sequence):
    def __init__(self, data_type, dataset, batch_size, char_vocab, fw_bigram_vocab, bw_bigram_vocab,
                 tag_vocab, shuffle=True):
        self.data = load_processed_data(data_type, dataset)
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.steps = int(np.ceil(self.data_size / self.batch_size))
        self.indices = np.arange(self.data_size)
        self.shuffle = shuffle

        self.char_vocab = char_vocab
        self.fw_bigram_vocab = fw_bigram_vocab
        self.bw_bigram_vocab = bw_bigram_vocab
        self.tag_vocab = tag_vocab

        self.n_class = len(self.tag_vocab)
        self.n_gaze = len(self.data[0].gaze_match)

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        return self.prepare_input(batch_indices)

    def prepare_input(self, batch_indices):
        max_token = np.max([len(self.data[index].tokens) for index in batch_indices])
        max_len = max_token + self.n_gaze * 2  # reserved for gaze node

        batch_char_ids, batch_fw_bigram_ids, batch_bw_bigram_ids, batch_adj_matrix = [], [], [], []
        batch_tag_ids = []
        for index in batch_indices:
            text_example = self.data[index]

            char_ids = [self.char_vocab.get(token, UNK) for token in text_example.tokens]
            batch_char_ids.append(char_ids)

            fw_bigram_ids = [self.fw_bigram_vocab.get(fw_bigram, UNK)
                             for fw_bigram in text_example.fw_bigrams]
            batch_fw_bigram_ids.append(fw_bigram_ids)

            bw_bigram_ids = [self.bw_bigram_vocab.get(bw_bigram, UNK)
                             for bw_bigram in text_example.bw_bigrams]
            batch_bw_bigram_ids.append(bw_bigram_ids)

            batch_adj_matrix.append(self.gen_adj_matrix(max_token=max_token,
                                                        n_token=len(text_example.tokens),
                                                        gaze_match=text_example.gaze_match))

            if text_example.tags:
                tag_ids = [self.tag_vocab[tag] for tag in text_example.tags]
                tag_ids = to_categorical(tag_ids, self.n_class).astype(int)
                batch_tag_ids.append(tag_ids)

        batch_gaze = np.tile(np.arange(self.n_gaze * 2), len(batch_indices))
        batch_gaze = np.reshape(batch_gaze, (-1, self.n_gaze * 2))
        batch_inputs = [pad_sequences_1d(batch_char_ids, max_len=max_len),
                        pad_sequences_1d(batch_fw_bigram_ids, max_len=max_len),
                        pad_sequences_1d(batch_bw_bigram_ids, max_len=max_len),
                        np.array(batch_adj_matrix),
                        batch_gaze]
        if not batch_tag_ids:
            return batch_inputs, None
        else:
            batch_tag_ids = pad_sequences_2d(batch_tag_ids, max_len_1=max_token,
                                             max_len_2=self.n_class)
            return batch_inputs, batch_tag_ids

    @staticmethod
    def gen_adj_matrix(max_token, n_token, gaze_match, init_value=1):
        n_gaze = len(gaze_match)
        n_node = max_token + n_gaze * 2
        adj_matrix = np.zeros(shape=((n_gaze + 1) * 2, n_node, n_node))

        for i in range(n_token):
            if i > 0:
                adj_matrix[0][i][i - 1] = init_value
            if i < n_token - 1:
                adj_matrix[1][i][i + 1] = init_value

        for gaze_id, matches in enumerate(gaze_match.values()):
            gaze_start_node = n_token + gaze_id * 2
            gaze_end_node = gaze_start_node + 1

            for match in matches:
                adj_matrix[gaze_id * 2 + 2][match[0]][gaze_start_node] = init_value
                adj_matrix[gaze_id * 2 + 2][gaze_end_node][match[1]] = init_value
                adj_matrix[gaze_id * 2 + 3][gaze_start_node][match[0]] = init_value
                adj_matrix[gaze_id * 2 + 3][match[1]][gaze_end_node] = init_value

                for i in range(match[0], match[1] + 1):
                    if i > match[0]:
                        adj_matrix[gaze_id * 2 + 2][i][i - 1] = init_value
                    if i < match[1]:
                        adj_matrix[gaze_id * 2 + 3][i][i + 1] = init_value

        return adj_matrix
