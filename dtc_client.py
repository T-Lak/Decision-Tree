import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import category_encoders as ce
import pandas as pd
import os

import calculation
from decisionTreeClassifier import DecisionTreeClassifier

files = ['car_evaluation.csv', 'iris.csv']


def start():
    print('[0] {}\n[1] {}'.format(files[0], files[1]))
    choice = int(input('choose a file: '))

    if choice == 0:
        col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        file = files[0]
    else:
        col_names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'class']
        file = files[1]

    df = pd.read_csv('./data/{}'.format(file))
    df.columns = col_names
    labels = sorted(df['class'].unique())
    df = ce.OrdinalEncoder(cols=list(df.columns[:-1])).fit_transform(df)
    x = df.drop(['class'], axis=1)
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    dtc = DecisionTreeClassifier(max_depth=3)
    dtc.train(x_train, y_train)
    y_pred_train = dtc.predict(x_train)
    y_pred_test = dtc.predict(x_test)

    tree = dtc.tree
    tree.make_dot_files()

    u_quit = ''
    while u_quit.lower() != 'quit':
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Analysis Options:\n{}\n{}\n{}\n{}'.format('[0] accuracy score', '[1] confusion matrix',
                                                         '[2] multilabel matrix', '[3] classification report'))
        option = int(input())
        os.system('cls' if os.name == 'nt' else 'clear')
        if option == 0:
            accuracy_score_train = calculation.accuracy_score(y_train, y_pred_train)
            accuracy_score_test = calculation.accuracy_score(y_test, y_pred_test)
            print('train set: ', accuracy_score_train)
            print('test set: ', accuracy_score_test)
        if option == 1:
            matrix_dict = calculation.build_matrix_dict(labels)

            lib_matrix = confusion_matrix(y_test, y_pred_test)
            own_matrix = calculation.confusion_matrix(y_true=y_test, y_pred=y_pred_test, matrix_dict=matrix_dict)

            print('lib matrix')
            print(lib_matrix)
            print('\nown matrix')
            print(np.matrix(own_matrix))
        if option == 2:
            m_dict = calculation.build_multilabel_matrix_dict(labels)
            multilabel_matrix_dict, multilabel_matrix = calculation.multilabel_matrix(y_true=y_test,
                                                                                      y_pred=y_pred_test,
                                                                                      matrix_dict=m_dict)

            for key, value in multilabel_matrix.items():
                print(key)
                print(np.matrix(value), '\n')
        if option == 3:
            m_dict = calculation.build_multilabel_matrix_dict(labels)
            multilabel_matrix_dict, multilabel_matrix = calculation.multilabel_matrix(y_true=y_test,
                                                                                      y_pred=y_pred_test,
                                                                                      matrix_dict=m_dict)
            report_dict = calculation.classification_report(multilabel_matrix_dict)
            print_classification_report_as_table(report_dict)

        u_quit = input('\ncontinue? [Enter/quit]: ')


def print_classification_report_as_table(report_dict):
    print('{:>10} {:>1} {:>15} {:>15}\n'.format('', 'precision', 'recall', 'f1-score'))
    for label, values in report_dict.items():
        print('{:>10} {:>9} {:>15} {:>15}'.format(label, values['precision'], values['recall'], values['f1-score']))


if __name__ == '__main__':
    start()


