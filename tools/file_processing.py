import os
import wget
import bz2
from gensim.models import KeyedVectors
from os import listdir
from os.path import isfile, join
from settings import WORD_EMBEDDING_MODELS_PATH, WORD_EMBEDDING_MODEL_NAME, WORD_EMBEDDING_MODEL_URL, ENCODING


def load_data(path_to_dir):
    """
    Text data loading

    :param path_to_dir: path to directory that contains files
    :return:            list of string
    """

    texts = []
    files = [file for file in listdir(path_to_dir)
             if isfile(join(path_to_dir, file)) and file.endswith(".txt")]

    for file in files:
        with open(join(path_to_dir, file), "r", encoding=ENCODING) as f:
            text = f.read()
            texts.append(text)

    return texts


def read_file(path_to_file, encoding=ENCODING):
    with open(path_to_file, encoding=encoding) as file:
        return file.read()


def load_embedding_model():
    if os.path.exists(WORD_EMBEDDING_MODELS_PATH + '/' + WORD_EMBEDDING_MODEL_NAME):
        embedding_model = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_MODELS_PATH + '/' + WORD_EMBEDDING_MODEL_NAME)
        return embedding_model
    else:
        print('Beginning file download with wget module')
        wget.download(WORD_EMBEDDING_MODEL_URL, WORD_EMBEDDING_MODELS_PATH)
        embedding_model_file = bz2.BZ2File(WORD_EMBEDDING_MODELS_PATH + '/' + WORD_EMBEDDING_MODEL_NAME + '.bz2', 'rb')

        try:
            model = embedding_model_file.read()
        finally:
            embedding_model_file.close()

        with open(WORD_EMBEDDING_MODELS_PATH + '/' + WORD_EMBEDDING_MODEL_NAME, 'wb') as new_file:
            new_file.write(model)
        embedding_model = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_MODELS_PATH + '/' + WORD_EMBEDDING_MODEL_NAME)
        return embedding_model


def write_txt_file(data, mode, file_path):
    with open(file_path, mode=mode, encoding=ENCODING) as new_file:
            new_file.write(data)


def recode(file_path, input_encoding):
    cleaned_text_cp1251 = read_file(file_path, encoding=input_encoding)
    byte_ENCODING = cleaned_text_cp1251.encode(encoding=ENCODING, errors='strict')
    text_ENCODING = byte_ENCODING.decode(encoding=ENCODING)
    write_txt_file(text_ENCODING, 'w', file_path)


def load_datasets(dir_path, encoding=ENCODING):
    train = read_file(dir_path + '/' + 'train_set.txt', encoding).split('\n')
    validate = read_file(dir_path + '/' + 'validate_set.txt', encoding).split('\n')
    test = read_file(dir_path + '/' + 'test_set.txt', encoding).split('\n')
    return train, validate, test

