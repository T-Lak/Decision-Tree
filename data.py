from joblib.numpy_pickle_utils import xrange


def class_data(sample, col_name=None, val=None):
    if col_name is None:
        total = sample['class'].count()
        return sample['class'].value_counts().to_dict(), total
    else:
        subset = sample.loc[sample[col_name] == val]
        total = subset['class'].count()
        return subset['class'].value_counts().to_dict(), total


def average_weight(table, col_name):
    attributes = sorted(table[col_name].unique())
    attr_list = []
    for i in xrange(len(attributes) - 1):
        attr_list.append((attributes[i] + attributes[i + 1]) / 2)
    return attr_list


def subsets_by_avg_weight(table, col_name, split_value):
    subset_1 = table.loc[table[col_name] < split_value]
    subset_2 = table.loc[table[col_name] > split_value]
    return [subset_1, subset_2]


def all_subsets(table, col_name):
    avg_weights = average_weight(table, col_name)
    a_dict = {}

    for weight in avg_weights:
        subset_1, subset_2 = subsets_by_avg_weight(table, col_name, weight)
        di1 = subset_1['class'].value_counts().to_dict()
        di2 = subset_2['class'].value_counts().to_dict()

        a_dict[weight] = []
        a_dict[weight] = []
        a_dict[weight].append([di1, sum(di1.values())])
        a_dict[weight].append([di2, sum(di2.values())])

    return a_dict, table['class'].count()
