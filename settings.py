import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATASET_DIR_PATH = os.path.join(PROJECT_ROOT, 'data', 'datasets')
RAW_TEXT_DIR_PATH = os.path.join(PROJECT_ROOT, 'data', 'texts', 'raw')
RAW_CORPUS_DIR_PATH = os.path.join(PROJECT_ROOT, 'data', 'texts', 'raw', 'corpus')
POS_TAGS_DIR_PATH = os.path.join(PROJECT_ROOT, 'data', 'doc')
TAGGED_TEXT_DIR_PATH = os.path.join(PROJECT_ROOT, 'data', 'texts', 'tagged')
WORDFORMS_DIR_PATH = os.path.join(PROJECT_ROOT, 'data', 'word_stock')
VECTORS_DIR_PATH = os.path.join(PROJECT_ROOT, 'data', 'word_stock', 'vectors')

WORD_EMBEDDING_MODELS_PATH = os.path.join(PROJECT_ROOT, 'models', 'word_embedding_models')
MODEL_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'models', 'model_checkpoints')

WORD_EMBEDDING_MODEL_URL = 'http://lang.org.ua/static/downloads/models/ubercorpus.lowercased.lemmatized.word2vec.300d.bz2'
WORD_EMBEDDING_MODEL_NAME = 'ubercorpus.lowercased.lemmatized.word2vec.300d'

TAG_TEXT_TOOL_FILE_PATH = os.path.join(PROJECT_ROOT, 'nlp_uk', 'src', 'main', 'groovy', 'org', 'nlp_uk', 'tools')
CLEAN_TEXT_TOOL_FILE_PATH = os.path.join(PROJECT_ROOT, 'nlp_uk', 'src', 'main', 'groovy', 'org', 'nlp_uk', 'other')

ENCODING='utf-8'

TIME_STEPS = 30
SKIP_STEPS = TIME_STEPS
BATCH_SIZE = 32
NUM_EPOCHS = 60
NUM_PREDICT = 30
