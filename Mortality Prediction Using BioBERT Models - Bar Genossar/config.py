import torch.nn as nn


class ConfigMain:
    DATA_FRAME = 'MORTALITY_TRAIN.csv'
    MAX_LEN = 96
    TRAIN_RATIO = 0.75
    TEXT_FEATURE = 'TEXT'
    MODEL_NAME = 'ACRCNN'
    MODELS_PATH = './/trained_models//' + MODEL_NAME + '//'
    EMBEDDINGS_VERSION = 'biobert_v1.1_pubmed'
    EMBEDDINGS_PATH = 'monologg/biobert_v1.1_pubmed'
    '''
    The possible values are : train_primary, train_sub, train_primary_sub, test
    '''
    TASK_TYPE = "test"


class ConfigPrimary:
    LOSS_FUNCTION = nn.NLLLoss()
    LABELS_NUM = 5
    DROPOUT = 0.0
    EPOCHS_NUM = 4
    THRESHOLD = 0.6

    HIDDEN_DIM_LSTM_val = 150
    HIDDEN_DIM_LSTM = [100, 250, 16]
    LINEAR_OUTPUT_DIM_val = 250
    LINEAR_OUTPUT_DIM = [150, 350, 21]
    LEARNING_RATE_val = 0.05
    LEARNING_RATE = [0.06, 0.15, 10]
    BATCH_SIZE_VAL = 32
    BATCH_SIZE = [48, 96, 7]
    MOMENTUM_val = 0.3
    MOMENTUM = [0, 0.9, 10]


class ConfigSubModel:
    LOSS_FUNCTION = nn.NLLLoss()
    LABELS_NUM = 2
    DROPOUT = 0.0
    HIDDEN_DIM_LSTM = [100, 250, 16]
    LINEAR_OUTPUT_DIM = [200, 350, 16]
    LEARNING_RATE = [0.01, 0.15, 15]
    EPOCHS_NUM = 5
    BATCH_SIZE = 32
    MOMENTUM = [0, 0.9, 10]
    THRESHOLD = 0
    BATCH_SIZE_VAL = 32





