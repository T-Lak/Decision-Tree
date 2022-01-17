from joblib.numpy_pickle_utils import xrange
from collections import deque

import calculation
import operator
from data import Data
from tree import Tree


class DecisionTreeClassifier:

    def __init__(self, data, test_set_size, max_depth=3):
        self.data = data
        self.tree = Tree()
        self.queue = deque()
        self.max_depth = max_depth
        self.train_set, self.test_set = self.data.split_into_train_and_test(test_set_size)

    def train(self):
        self.init_training()
        count = 0

        while (count < self.max_depth) & (len(self.queue) > 0):
            node = self.queue.popleft()
            for edge in node.edges:
                subset = edge.table.drop([node.label], axis=1)
                data_dict = self.find_best_split_feature(subset)
                n = self.tree.build_node(self.data, data_dict, subset)
                if data_dict['entropy'] > 0.0:
                    self.queue.append(n)
                    node.children.append(n)
                edge.child = n
            count += 1

    def init_training(self):
        data_dict = self.find_best_split_feature(self.train_set)
        root_node = self.tree.build_node(self.data, data_dict, self.train_set)
        self.tree.root_node = root_node
        self.queue.append(root_node)

    def find_best_split_feature(self, table):
        class_data, subset_size = self.data.get_class_data(table)
        entropy = calculation.calc_entropy(class_data, subset_size)
        feature_data = {'key': None, 'split_value': None, 'entropy': entropy, 'samples': subset_size,
                        'value': None, 'class_var': None}
        values = {}

        for column in table.columns[:-1]:
            subset, size = self.data.get_all_subsets_by_avg_weight(table, column)
            a_w_entropy = calculation.calc_information_gain(subset, size, entropy)
            values[column] = max(a_w_entropy.items(), key=operator.itemgetter(1))

        tup = max(values.items(), key=lambda sub: sub[1][1])
        feature_data['key'] = tup[0]
        feature_data['split_value'] = tup[1][0]
        feature_data['value'] = list(class_data.values())
        feature_data['class_var'] = max(class_data.items(), key=operator.itemgetter(1))[0]
        return feature_data

    def predict(self):
        y_pred = []
        for index, row in self.test_set.iterrows():
            node = self.tree.root_node
            result = self.bla(node, row)
            y_pred.append(result)
        count = 0
        for index in xrange(len(y_pred)):
            if y_pred[index] == self.test_set['class'].to_list()[index]:
                count += 1
        return count / len(y_pred)

    def bla(self, node, data_dict):
        if not node.children:
            return node.class_var
        value = data_dict[node.label]
        split_value = node.split_value
        if value <= split_value:
            return self.bla(node.edges[1].child, data_dict)
        else:
            return self.bla(node.edges[0].child, data_dict)


dtc = DecisionTreeClassifier(Data(), 0.33, 3)
dtc.train()
val = dtc.predict()
print(val)
tree = dtc.tree
tree.visualize()
