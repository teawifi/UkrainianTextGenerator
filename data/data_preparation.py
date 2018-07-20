from data.vocabulary import Vocabulary
from tools.parser import Parser
from tools.file_processing import load_data, write_txt_file, read_file, recode
from tools.wrappers import LanguageToolWrapper
from settings import RAW_CORPUS_DIR_PATH, CLEAN_TEXT_TOOL_FILE_PATH, TAG_TEXT_TOOL_FILE_PATH
import numpy as np


class DataPreparation:
    def __init__(self, word_embedding_model, num_time_steps=30,
                 skip_steps=30, train_size=0.6, test_size=0.2):
        self.model = word_embedding_model
        self.vocabulary = Vocabulary()
        self.train_size = train_size
        self.test_size = test_size
        self.num_time_steps = num_time_steps       
        self.skip_steps = skip_steps
        self.lemma_vector_size = self.model.vector_size
        self.pos_binary_vector_size = 0

    def execute(self, tagged_text_file_path, raw_text_dir_path):
        raw_text = load_data(raw_text_dir_path)
        # save raw_text
        write_txt_file(data=' '.join(map(str, raw_text)), mode='w', file_path=RAW_CORPUS_DIR_PATH + '/corpus.txt')
        LanguageToolWrapper.run_clean_text_utility(CLEAN_TEXT_TOOL_FILE_PATH + '/',
                                                    input_directory=RAW_CORPUS_DIR_PATH)
        
        recode(RAW_CORPUS_DIR_PATH + '/good' + '/corpus.txt', input_encoding='cp1251')
        
        LanguageToolWrapper.run_tag_text_utility(TAG_TEXT_TOOL_FILE_PATH + '/',
                                                  input_file=RAW_CORPUS_DIR_PATH + '/good' + '/corpus.txt',
                                                  output_file=tagged_text_file_path)

        index2word_set = set(self.model.wv.index2word)
        parser = Parser()
        data = parser.execute(index2word_set, tagged_text_file_path)
        grammatical_values = self.collect_grammatical_values(data)
        train_set, validate_set, test_set = self.train_validate_test_split(data)
        self.pos_binary_vector_size = len(grammatical_values)
        self.vocabulary.build(data, grammatical_values)

        return train_set, validate_set, test_set

    def to_categorical(self, words):
        y = np.zeros(shape=(self.num_time_steps, self.vocabulary.size))

        for i, word in enumerate(words):
            y[i, self.vocabulary.wordforms_lemma_gramvalue_to_idx[word]] = 1

        return y

    def train_validate_test_split(self, data):
        return data[: int(self.train_size * len(data))], \
                   data[int(self.train_size * len(data)): int(self.train_size * len(data)) + int(self.test_size * len(data))], \
                   data[int(self.train_size * len(data)) + int(self.test_size * len(data)):]

    def transform_wordform_to_vectors(self, data):
        '''
        transform word forms to lemma embedding and POS binary vectors

        :param data:
        :return:
        '''
        lemma_vector = []
        pos_binary_vector = []

        # item[1] - lemma
        # item[2:] - pos and additional tags
        for item in data:
            lemma_vector.append(self.model[item[1].strip(' ').lower()])
            pos_vector = np.zeros(shape=len(self.vocabulary.pos_tags))
            for tag in item[2:]:
                if tag in self.vocabulary.pos_tags_indices:
                    index = self.vocabulary.pos_tags_indices[tag]
                    pos_vector[index] = 1
                else:
                    print('Tag does not exist: ', tag)
            pos_binary_vector.append(pos_vector)

        return lemma_vector, pos_binary_vector

    def collect_grammatical_values(self, data):
        pos_tags = []
        for item in data:
            for tag in item[2:]:
                pos_tags.append(tag.strip(' '))

        return sorted(set(pos_tags))

