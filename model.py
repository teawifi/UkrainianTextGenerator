from keras.models import Model
from keras.layers import Input, Dense, LSTM, TimeDistributed, Activation, concatenate
from keras.callbacks import ModelCheckpoint

import numpy as np


class TextGenerator:
    """
    Model description
    """
    def __init__(self,num_time_steps, lemma_vector_size, pos_tag_vector_size, dense_units, hidden_size=500,
                 dropout=0.2, batch_size=20):
        self.model = None
        self.batch_size = batch_size
        self.pos_tag_vector_size = pos_tag_vector_size

        self.build(num_time_steps, lemma_vector_size, pos_tag_vector_size, dense_units, hidden_size, dropout)

    def build(self, num_time_steps, lemma_vector_size, pos_tag_vector_size, dense_units, hidden_size, dropout):
        lemma = Input(shape=(num_time_steps, lemma_vector_size), name='lemma_input')
        grammatical_value = Input(shape=(num_time_steps, pos_tag_vector_size), name='pos_tag_input')

        grammar_dense_nb_units_1 = pos_tag_vector_size
        grammar_dense_nb_units_2 = int(grammar_dense_nb_units_1/ 2)

        grammatical_layer1 = Dense(grammar_dense_nb_units_1, activation='relu')(grammatical_value)
        grammatical_layer2 = Dense(grammar_dense_nb_units_2, activation='relu')(grammatical_layer1)

        lstm_input = concatenate([lemma, grammatical_layer2], name='LSTM_input', axis=-1)
        lstm_1 = LSTM(hidden_size, dropout=dropout, return_sequences=True, name='LSTM_1')(lstm_input)
        lstm_2 = LSTM(hidden_size, dropout=dropout, return_sequences=True, name='LSTM_2')(lstm_1)

        dense_layer = TimeDistributed(Dense(dense_units))(lstm_2)
        output = Activation('softmax')(dense_layer)

        self.model = Model(inputs=[lemma, grammatical_value], outputs=[output])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

        # summarize layers
        print(self.model.summary())

    def train(self, x_train, y_train, x_validate, y_validate, model_checkpoint_path, epochs=60):
        checkpointer = ModelCheckpoint(filepath=model_checkpoint_path + '/model-{epoch:02d}.hdf5', verbose=1)
        self.model.fit(x=x_train, y=y_train,
                       batch_size=self.batch_size,
                       epochs=epochs,
                       verbose=1,
                       callbacks=[checkpointer],
                       validation_data=(x_validate, y_validate))

        self.model.save(model_checkpoint_path + "/final_model.hdf5")

    def train_generator(self, generator, steps_per_epoch, validation_data, validation_steps, model_checkpoint_path, epochs=60):
        checkpointer = ModelCheckpoint(filepath=model_checkpoint_path + '/model-{epoch:02d}.hdf5', verbose=1)

        self.model.fit_generator(generator=generator,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks = [checkpointer])

        self.model.save(model_checkpoint_path + "/final_model.hdf5")

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=self.batch_size, verbose=1)

    def evaluate_generator(self, generator, steps, verbose=1):
        return self.model.evaluate_generator(generator, steps, verbose)

    def predict(self, data):
        return self.model.predict(data)

    def predict_generator(self, generator, steps=1):
        return self.model.predict_generator(generator, steps)

    def generate(self, data, data_preparation, num_time_steps, batch_size=1, start_index=0, num_predict=10):
        """

        :param data:
        :param num_time_steps:
        :param start_index:
        :param num_predict:
        :return:
        """

        vocabulary = data_preparation.vocabulary
        data_length = len(data)
        lemmas_sample = np.zeros(shape=(batch_size, num_time_steps + num_predict, data_preparation.lemma_vector_size))
        gramm_values_sample = np.empty(shape=(batch_size, num_time_steps + num_predict, data_preparation.pos_binary_vector_size))

        if start_index + num_time_steps + num_predict <= data_length:
            lemmas_sample[0, 0: num_time_steps], gramm_values_sample[0, 0: num_time_steps] = \
                data_preparation.transform_wordform_to_vectors(data[start_index: start_index + num_time_steps])

            expected_values = ''
            for word in data[start_index: start_index + num_time_steps + num_predict]:
                expected_values += ' ' + word[0]

            predicted_values = ''
            for word in data[start_index: start_index + num_time_steps]:
                predicted_values += ' ' + word[0]

            for i in range(0, num_predict):
                prediction = self.predict([lemmas_sample[:, i: i + num_time_steps, :], gramm_values_sample[:, i: i + num_time_steps, :]])
                predicted_word_index = np.argmax(prediction[:, num_time_steps - 1, :])

                if predicted_word_index < vocabulary.size:
                    predicted_word = vocabulary.idx_to_wordforms_lemma_gramvalue[predicted_word_index]
                    predicted_values += " " + predicted_word.split('|')[0]

                    lemmas, gramm_values = data_preparation.transform_wordform_to_vectors([predicted_word.split('|')])
                    lemmas_sample[:, num_time_steps + i, :] = lemmas
                    gramm_values_sample[:, num_time_steps + i, :] = gramm_values

            print('Expected text: ', expected_values)
            print('Predicted text: ', predicted_values)
        else:
            print('Out of range of the data.'
                  'Please, enter lower value of variables: start_index or num_time_steps or num_predict.'
                  'It is necessary that the next condition is executed: '
                  'start_index + num_time_steps + num_predict <= data_length')
