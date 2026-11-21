import os
import random
import numpy as np

# Reproducibility Seed
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Dataset specifications
DATASET_PATH = "indonlp/indonlu"
DATASET_NAME = "terma"

# Label
LABEL_LIST = ['I-SENTIMENT', 'O', 'I-ASPECT', 'B-SENTIMENT', 'B-ASPECT']
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}

# Model Hyperparameters
MODEL_NAME = "indobenchmark/indobert-base-p1"
FALLBACK_MODEL_NAME = "indolem/indobert-base-uncased"
MAX_LEN = 72
BATCH_SIZE = 8
LEARNING_RATE = [1e-5, 2e-5]
EPOCHS = 3
WEIGHT_DECAY = [0.1, 0.01]

# CRF Hyperparameter Search Space (on Validation)
CRF_C1_VALUES = [0.01, 0.1, 1.0]
CRF_C2_VALUES = [0.01, 0.1, 1.0]
CRF_ALGORITHM = "lbfgs"
CRF_MAX_ITERATIONS = 100
