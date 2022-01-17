from sklearn.model_selection import train_test_split
from joblib.numpy_pickle_utils import xrange
import category_encoders as ce
import pandas as pd


class Data:

    def __init__(self):
        col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        self.df = pd.read_csv('car_evaluation.csv')
        # col_names = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'class']
        # self.df = pd.read_csv('iris.csv')
        self.orig_data = self.df
        self.df.columns = col_names
        self.encoder = ce.OrdinalEncoder(cols=col_names[:-1])
        self.df = self.encoder.fit_transform(self.df)

    def split_into_train_and_test(self, test_size):
        return train_test_split(self.df, test_size=test_size, random_state=42)

    def get_class_data(self, sample, col_name=None, val=None):
        if col_name is None:
            total = sample['class'].count()
            return sample['class'].value_counts().to_dict(), total
        else:
            subset = sample.loc[sample[col_name] == val]
            total = subset['class'].count()
            return subset['class'].value_counts().to_dict(), total

    def calc_avg_weight(self, table, col_name):
        attributes = sorted(table[col_name].unique())
        attr_list = []
        for i in xrange(len(attributes) - 1):
            attr_list.append((attributes[i] + attributes[i + 1]) / 2)
        return attr_list

    def get_all_subsets_by_avg_weight_1(self, table, col_name, split_value):
        subset_1 = table.loc[table[col_name] < split_value]
        subset_2 = table.loc[table[col_name] > split_value]
        return [subset_1, subset_2]

    def get_all_subsets_by_avg_weight(self, table, col_name):
        avg_weights = self.calc_avg_weight(table, col_name)
        a_dict = {}

        for weight in avg_weights:
            subset_1, subset_2 = self.get_all_subsets_by_avg_weight_1(table, col_name, weight)
            di1 = subset_1['class'].value_counts().to_dict()
            di2 = subset_2['class'].value_counts().to_dict()

            a_dict[weight] = []
            a_dict[weight] = []
            a_dict[weight].append([di1, sum(di1.values())])
            a_dict[weight].append([di2, sum(di2.values())])

        return a_dict, table['class'].count()

