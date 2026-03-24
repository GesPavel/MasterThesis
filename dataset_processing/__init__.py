import os

RAW_DATA_FOLDER = 'raw_data'
if not os.path.exists(RAW_DATA_FOLDER):
    raise FileNotFoundError('Raw data folder does not exist. Create it, name it raw_data and fill it.')
DATASET_PICKLE_PATH = os.path.join(RAW_DATA_FOLDER, 'MSL40.pkl')

PROCESSED_DATA_FOLDER = 'processed_data'
if not os.path.exists(PROCESSED_DATA_FOLDER):
    os.makedirs(PROCESSED_DATA_FOLDER)
DATASET_OVERVIEW_FOLDER = os.path.join(PROCESSED_DATA_FOLDER, 'overview')
if not os.path.exists(DATASET_OVERVIEW_FOLDER):
    os.makedirs(DATASET_OVERVIEW_FOLDER)
SPLIT_DATASET_FOLDER = os.path.join(DATASET_OVERVIEW_FOLDER, 'split')
if not os.path.exists(SPLIT_DATASET_FOLDER):
    os.makedirs(SPLIT_DATASET_FOLDER)

FILENAME_NOVELTY = 'novelty_test_set.csv'
FILENAME_RANDOM = 'random_test_set.csv'
FILENAME_TRAIN = 'train_set.csv'
