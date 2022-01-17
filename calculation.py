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
