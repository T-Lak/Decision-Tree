"""
Configuration file for global constants and settings.
"""

# === Folders ===
DATASET_FOLDER_PATH = "data/datasets"
RESULTS_FOLDER_PATH = "data/results"

# === Algorithm Parameters ===

## Data Splitting
TEST_SIZE = 0.33  # Proportion of the dataset used for testing
RANDOM_STATE = 42  # Ensures reproducibility

## Classifier Settings
MAX_DEPTH = 4  # Maximum depth of the decision tree