import argparse
import os

from sklearn.model_selection import train_test_split
import category_encoders as ce
import pandas as pd

from config import DATASET_FOLDER_PATH
from metrics import calculation
from algo.decisionTreeClassifier import DecisionTreeClassifier
from storage.metrics.metrics_saver import save_results, save_multilabel_matrix, save_matrix_to_csv


def main(dataset_filename):
    dataset_path = os.path.join(DATASET_FOLDER_PATH, dataset_filename)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    df = pd.read_csv(dataset_path)
    columns_names = df.columns.tolist()
    columns_names[-1] = 'class'
    df.columns = columns_names
    labels = sorted(df['class'].unique())
    df = ce.OrdinalEncoder(cols=list(df.columns[:-1])).fit_transform(df)
    x = df.drop(['class'], axis=1)
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    dtc = DecisionTreeClassifier(max_depth=4)
    dtc.train(x_train, y_train)
    y_pred_train = dtc.predict(x_train)
    y_pred_test = dtc.predict(x_test)

    tree = dtc.tree
    tree.make_dot_files()

    accuracy_score_train = calculation.accuracy_score(y_train, y_pred_train)
    accuracy_score_test = calculation.accuracy_score(y_test, y_pred_test)

    matrix_dict = calculation.build_matrix_dict(labels)
    conf_matrix = calculation.confusion_matrix(y_true=y_test, y_pred=y_pred_test, matrix_dict=matrix_dict)

    m_dict = calculation.build_multilabel_matrix_dict(labels)
    multilabel_matrix_dict, multilabel_matrix = calculation.multilabel_matrix(y_true=y_test,
                                                                              y_pred=y_pred_test,
                                                                              matrix_dict=m_dict)
    report_dict = calculation.classification_report(multilabel_matrix_dict)

    results = {
        "dataset": dataset_filename,
        "accuracy_train": accuracy_score_train,
        "accuracy_test": accuracy_score_test,
        "classification_report": report_dict
    }

    save_results(results)
    save_matrix_to_csv(conf_matrix, labels, "matrix.csv")
    save_multilabel_matrix(multilabel_matrix, "multilabel.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the classification program with a specific dataset file.")
    parser.add_argument("dataset_filename", type=str, help="The name of the dataset file (e.g., iris.csv)")

    args = parser.parse_args()
    main(args.dataset_filename)