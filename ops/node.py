class Node(object):

    def __init__(self, input_nodes=[]):
        """
        Define the inputs to this node and create variables to hold the output, gradients, and output nodes.
        :param input_nodes: The nodes providing input to this node
        """
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in self.input_nodes:
            node.output_nodes.append(self)

        self.output = None
        self.gradients = {}

    def compute(self, output=None):
        raise NotImplementedError

    def backpass(self):
        raise NotImplementedError
