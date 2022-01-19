from edge import Edge


class Node:

    def __init__(self, label, entropy, split_value, samples, value, class_var):
        self.label = label
        self.entropy = entropy
        self.split_value = split_value
        self.samples = samples
        self.value = value
        self.class_var = class_var
        self.edges = []
        self.children = []

    def add_edges(self, subsets):
        self.edges.append(Edge('True', subsets[1], self))
        self.edges.append(Edge('False', subsets[0], self))

    def get_info(self):
        return "{label} ≤ {split}\nentropy = {entropy}\nsamples = {samples}\nvalue = {value}\nclass = {classV}".format(
            label=self.label, split=self.split_value, entropy='%.2f' % self.entropy, samples=self.samples,
            value=self.value, classV=self.class_var)

    def get_info_without_label(self):
        return "entropy = {entropy}\nsamples = {samples}\nvalue = {value}\nclass = {classV}".format(
            entropy='%.2f' % self.entropy, samples=self.samples, value=self.value, classV=self.class_var)
