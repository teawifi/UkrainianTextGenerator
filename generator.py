from model import TextGenerator
from data.data_preparation import DataPreparation
from tools.batch_generator import BatchGenerator
from settings import WORD_EMBEDDING_MODELS_PATH, MODEL_CHECKPOINT_PATH, POS_TAGS_DIR_PATH, WORD_EMBEDDING_MODEL_URL,\
    WORD_EMBEDDING_MODEL_NAME, RAW_TEXT_DIR_PATH, TAGGED_TEXT_DIR_PATH, TIME_STEPS, SKIP_STEPS, BATCH_SIZE, NUM_EPOCHS,\
    NUM_PREDICT
from tools.file_processing import load_embedding_model

# load Word2Vec model
embedding_model = load_embedding_model()
preparation = DataPreparation(word_embedding_model=embedding_model)

print('Performance of data preparation...')

train, validate, test = preparation.execute(tagged_text_file_path=TAGGED_TEXT_DIR_PATH + '\\tagged_input.txt',
                                                        raw_text_dir_path=RAW_TEXT_DIR_PATH)

train_batch_generator = BatchGenerator(data=train,
                                        preparation=preparation,
                                        num_time_steps=TIME_STEPS,
                                        skip_steps=SKIP_STEPS,
                                        batch_size=BATCH_SIZE)
validate_batch_generator = BatchGenerator(data=validate,
                                        preparation=preparation,
                                        num_time_steps=TIME_STEPS,
                                        skip_steps=SKIP_STEPS,
                                        batch_size=BATCH_SIZE)

print('Creating the language model...')
model = TextGenerator(num_time_steps=TIME_STEPS,
                      lemma_vector_size=preparation.lemma_vector_size,
                      pos_tag_vector_size=preparation.pos_binary_vector_size,
                      dense_units=preparation.vocabulary.size)

print('Model training...')
model.train_generator(generator=train_batch_generator.generate(),
                      steps_per_epoch=len(train)//BATCH_SIZE,
                      validation_data=validate_batch_generator.generate(),
                      validation_steps=len(validate)//BATCH_SIZE,
                      epochs=NUM_EPOCHS,
                      model_checkpoint_path=MODEL_CHECKPOINT_PATH)

print('Model evaluation...')
test_batch_generator = BatchGenerator(data=test,
                                        preparation=preparation,
                                        num_time_steps=TIME_STEPS,
                                        skip_steps=SKIP_STEPS,
                                        batch_size=BATCH_SIZE)
result = model.evaluate_generator(test_batch_generator.generate(), steps=len(test)//BATCH_SIZE)
print('Result: ', result)

print('Generate text from a train data...')
model.generate(train, num_time_steps=TIME_STEPS, data_preparation=preparation, num_predict=NUM_PREDICT)

print('Generate text from a test data...')
model.generate(test, num_time_steps=TIME_STEPS, data_preparation=preparation, num_predict=NUM_PREDICT)

