import json
import os

import pandas as pd

from config import RESULTS_FOLDER_PATH


def save_results(results):
    file_path = os.path.join(RESULTS_FOLDER_PATH, "metrics.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(results)
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)


def save_matrix_to_csv(matrix, labels, filename):
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(os.path.join(RESULTS_FOLDER_PATH, filename), index=True)


def save_multilabel_matrix(multilabel_matrix, filename):
    for label, matrix in multilabel_matrix.items():
        df = pd.DataFrame(matrix, index=["Pred_Pos", "Pred_Neg"], columns=["True_Pos", "True_Neg"])
        file_path = f"{filename}_{label}.csv"
        df.to_csv(os.path.join(RESULTS_FOLDER_PATH, file_path), index=True)