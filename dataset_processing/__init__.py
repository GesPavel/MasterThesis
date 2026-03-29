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
SPLIT_DATASET_FOLDER = os.path.join(PROCESSED_DATA_FOLDER, 'split')
if not os.path.exists(SPLIT_DATASET_FOLDER):
    os.makedirs(SPLIT_DATASET_FOLDER)

FILENAME_NOVELTY = 'novelty_test_set.csv'
FILENAME_RANDOM = 'random_test_set.csv'
FILENAME_TRAIN = 'train_set.csv'

OUTPUT_FOLDER = os.path.join(PROCESSED_DATA_FOLDER, 'output')
HMM_FOLDER_NAME = 'hmm'
EVALUATION_FOLDER_NAME = 'evaluation'

FILENAME_GROUND_TRUTH_PAIRS = 'ground_truth_pairs.csv'
FILENAME_LOGISTICAL_REGRESSION_MODEL = "logreg_model.joblib"
FILENAME_XGBOOST_MODEL = "xgboost_model.joblib"


# Acceptable types: 'train', 'random_test', 'novelty_test'
def get_dataset_filepath(rank, type):
    dataset_folder = os.path.join(SPLIT_DATASET_FOLDER, rank)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        print(f"Created dataset folder for rank {rank} at {dataset_folder}")
    if type == 'novelty':
        return os.path.join(dataset_folder, FILENAME_NOVELTY)
    elif type == 'random':
        return os.path.join(dataset_folder, FILENAME_RANDOM)
    elif type == 'train':
        return os.path.join(dataset_folder, FILENAME_TRAIN)
    else:
        raise ValueError('Type must be one of novelty, random, train.')


def get_intermediate_output_path(rank, method, type):
    output_folder_for_rank = os.path.join(OUTPUT_FOLDER, rank)
    if not os.path.exists(output_folder_for_rank):
        os.makedirs(output_folder_for_rank)
    if method == 'hmm':
        method_folder = os.path.join(output_folder_for_rank, HMM_FOLDER_NAME)
    else:
        raise ValueError('Only HMM is supported for method.')
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    if type == 'pairs':
        return os.path.join(method_folder, FILENAME_GROUND_TRUTH_PAIRS)
    elif type == 'logreg':
        return os.path.join(method_folder, FILENAME_LOGISTICAL_REGRESSION_MODEL)
    elif type == 'xgboost':
        return os.path.join(method_folder, FILENAME_XGBOOST_MODEL)
    else:
        raise ValueError(
            'Only ground truth pairs and logistical regression model are currently supported for output path')


def get_evaluation_output_dir(rank, method='hmm'):
    output_folder_for_rank = os.path.join(OUTPUT_FOLDER, rank)
    if not os.path.exists(output_folder_for_rank):
        os.makedirs(output_folder_for_rank)

    if method == 'hmm':
        method_folder = os.path.join(output_folder_for_rank, HMM_FOLDER_NAME)
    else:
        raise ValueError('Only HMM is supported for method.')

    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    evaluation_folder = os.path.join(method_folder, EVALUATION_FOLDER_NAME)
    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)

    return evaluation_folder

