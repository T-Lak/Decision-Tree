import math


def calc_entropy(dictionary, total):
    entropy = 0
    for entry in dictionary:
        if dictionary[entry] == 0:
            continue
        entropy += -((dictionary[entry] / total) * math.log2(dictionary[entry] / total))

    return entropy


def calc_information_gain(dictionary, total, parent_entropy):
    a_w_entropies = {}
    for key, val in dictionary.items():
        a_w_entropy = 0
        for a_list in val:
            entropy = calc_entropy(a_list[0], a_list[1])
            a_w_entropy += ((a_list[1] / total) * entropy)
        a_w_entropies[key] = parent_entropy - a_w_entropy
    return a_w_entropies


def accuracy_score(y_true, y_pred):
    count = 0
    total = len(y_pred)
    for idx in range(total):
        if y_pred[idx] == y_true.to_list()[idx]:
            count += 1
    return count / total


def classification_report(matrix_dict):
    report_dict = {}
    for label, value in matrix_dict.items():
        report_dict[label] = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}.copy()
        if (value['tp'] == 0) | (value['fp'] == 0) | (value['fn'] == 0):
            continue
        report_dict[label]['precision'] = round(value['tp'] / (value['tp'] + value['fp']), 2)
        report_dict[label]['recall'] = round(value['tp'] / (value['tp'] + value['fn']), 2)
        report_dict[label]['f1-score'] = round(value['tp'] / (value['tp'] + 0.5 * (value['fp'] + value['fn'])), 2)
    return report_dict


def confusion_matrix(y_true, y_pred, matrix_dict):
    y_true = y_true.to_list()
    total = len(y_pred)
    matrix = []
    for idx in range(total):
        if y_pred[idx] == y_true[idx]:
            matrix_dict[y_pred[idx]][y_pred[idx]] += 1
        if y_pred[idx] != y_true[idx]:
            matrix_dict[y_true[idx]][y_pred[idx]] += 1
    for di in matrix_dict.values():
        matrix.append(list(di.values()))
    return matrix


def multilabel_matrix(y_true, y_pred, matrix_dict):
    y_true = y_true.to_list()
    total = len(y_pred)
    matrix = {}
    for label, value in matrix_dict.items():
        for idx in range(total):
            if (y_pred[idx] == label) & (y_true[idx] == label):
                matrix_dict[label]['tp'] += 1
            if (y_pred[idx] != label) & (y_true[idx] == label):
                matrix_dict[label]['fn'] += 1
            if (y_pred[idx] == label) & (y_true[idx] != label):
                matrix_dict[label]['fp'] += 1
            if (y_pred[idx] != label) & (y_true[idx] != label):
                matrix_dict[label]['tn'] += 1
        matrix[label] = [[matrix_dict[label]['tp'], matrix_dict[label]['fn']],
                         [matrix_dict[label]['fp'], matrix_dict[label]['tn']]]
    return matrix_dict, matrix


def build_matrix_dict(labels):
    matrix_dict = {}
    for act_label in labels:
        matrix_dict[act_label] = {}
        for pred_label in labels:
            matrix_dict[act_label][pred_label] = 0
    return matrix_dict


def build_multilabel_matrix_dict(labels):
    matrix_dict = {}
    values = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for label in labels:
        matrix_dict[label] = values.copy()
    return matrix_dict
