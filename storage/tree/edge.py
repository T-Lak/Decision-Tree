class Edge:

    def __init__(self, label=None, table=None, parent=None, child=None):
        self.label = label
        self.table = table
        self.parent = parent
        self.child = child

    def print_edge(self):
        return self.label, self.parent.label, self.child.label



