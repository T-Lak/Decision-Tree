from graphviz import Digraph
from node import Node
import data


class Tree:

    def __init__(self):
        self.root_node = None
        self.edges = []
        self.dot_graph = Digraph()

    def add_edge(self, edge):
        self.edges.append(edge)

    def set_root_node(self, node):
        self.root_node = node

    def build_node(self, data_dict, subset):
        node = Node(data_dict['key'], data_dict['entropy'],
                    data_dict['split_value'], data_dict['samples'],
                    data_dict['value'], data_dict['class_var']
                    )
        d = data.subsets_by_avg_weight(subset, data_dict['key'], data_dict['split_value'])
        node.add_edges(d)
        return node

    def make_dot_files(self):
        self.dot_graph.attr('node', fontsize='10', shape='box', style='rounded, filled', fillcolor='lightblue')
        self.dot_graph.attr('edge', fontsize='10')
        self.build_dot_tree(self.root_node)
        self.dot_graph.render('./data/Decision_Tree.gv', view=False)

    def build_dot_tree(self, node):
        for edge in node.edges:
            if edge.child is None:
                continue

            parent = node.get_info()
            child = edge.child.get_info() if len(edge.child.children) > 0 else edge.child.get_info_without_label()
            self.dot_graph.edge(parent, child, str(edge.label))
            self.dot_graph.node(parent)
            self.dot_graph.node(child)
            self.build_dot_tree(edge.child)
