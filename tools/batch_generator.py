import numpy as np


class BatchGenerator:
    def __init__(self, data, preparation, num_time_steps=30, skip_steps=30, batch_size=32):
        self.data = data
        self.num_time_steps = num_time_steps
        self.skip_steps = skip_steps
        self.batch_size = batch_size
        self.current_idx = self.num_time_steps
        self.preparation = preparation

    def generate(self):
        x_lemma = np.zeros(shape=(self.batch_size, self.num_time_steps, self.preparation.lemma_vector_size))
        x_pos_tag = np.empty(shape=(self.batch_size, self.num_time_steps, self.preparation.pos_binary_vector_size))
        y = np.zeros(shape=(self.batch_size, self.num_time_steps, self.preparation.vocabulary.size))

        while True:
            for i in range(self.batch_size):
                if self.current_idx + 1 >= len(self.data):
                    self.current_idx = self.num_time_steps
                x_lemma[i, :], x_pos_tag[i, :] = \
                    self.preparation.transform_wordform_to_vectors(self.data[self.current_idx - self.num_time_steps: self.current_idx])
                next_words = self.data[self.current_idx - self.num_time_steps + 1: self.current_idx + 1]
                y[i, :, :] = self.preparation.to_categorical(['|'.join(map(str, word)) for word in next_words])
                self.current_idx += self.skip_steps
            yield [x_lemma, x_pos_tag], y
