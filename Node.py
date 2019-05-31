


class Node(object):

    def __init__(self, input_nodes = []):

        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in self.input_nodes:
            node.output_nodes.append(self)

        self.output = None

        self.gradients = {}


    def compute(self, output = None):

        raise NotImplementedError


    def backpass(self):

        raise NotImplementedError




