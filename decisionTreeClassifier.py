from collections import deque

import calculation
import operator
import data

from tree import Tree


class DecisionTreeClassifier:

    def __init__(self, max_depth=3, criterion='entropy'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = Tree()

    def train(self, x, y):
        x.insert(len(x.columns), 'class', y)
        data_dict = self.find_best_split_feature(x)
        root = self.tree.build_node(data_dict, x)
        self.tree.root_node = root
        queue = deque()
        queue.append(root)
        self.build(queue)

    def build(self, queue, count=0):
        if (count == self.max_depth) | (len(queue) == 0):
            return
        node = queue.popleft()
        for edge in node.edges:
            subset = edge.table.drop([node.label], axis=1)
            data_dict = self.find_best_split_feature(subset)
            n = self.tree.build_node(data_dict, subset)
            if data_dict['entropy'] > 0.0:
                queue.append(n)
                node.children.append(n)
            edge.child = n
        self.build(queue, count + 1)

    def find_best_split_feature(self, table):
        class_data, subset_size = data.class_data(table)
        entropy = calculation.calc_entropy(class_data, subset_size)
        feature_data = {
            'key': None, 'split_value': None, 'entropy': entropy,
            'samples': subset_size, 'value': None, 'class_var': None
        }
        values = {}

        for column in table.columns[:-1]:
            subset, size = data.all_subsets(table, column)
            a_w_entropy = calculation.calc_information_gain(subset, size, entropy)
            values[column] = max(a_w_entropy.items(), key=operator.itemgetter(1))

        tup = max(values.items(), key=lambda sub: sub[1][1])
        feature_data['key'] = tup[0]
        feature_data['split_value'] = tup[1][0]
        feature_data['value'] = list(class_data.values())
        feature_data['class_var'] = max(class_data.items(), key=operator.itemgetter(1))[0]
        return feature_data

    def predict(self, subset):
        y_pred = []
        for index, row in subset.iterrows():
            node = self.tree.root_node
            result = self.make_prediction(node, row)
            y_pred.append(result)
        return y_pred

    def make_prediction(self, node, subset_row):
        if not node.children:
            return node.class_var
        value = subset_row[node.label]
        split_value = node.split_value
        if value <= split_value:
            return self.make_prediction(node.edges[1].child, subset_row)
        else:
            return self.make_prediction(node.edges[0].child, subset_row)
