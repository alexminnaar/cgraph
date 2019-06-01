from ops.node import Node
import numpy as np


class Sigmoid(Node):

    def __init__(self, node):
        Node.__init__(self, [node])

    def compute(self):
        """
        Compute sigmoid activation based on input node.
        """
        input_value = self.input_nodes[0].output
        self.output = 1. / (1. + np.exp(-input_value))

    def backpass(self):
        """
        Backpropagate gradients to input node.
        """
        # clear gradients for input nodes
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        # backpropagate gradients to input nodes which is also a function of gradients from output nodes
        for node in self.output_nodes:
            grad_cost = node.gradients[self]
            sigmoid = self.output
            self.gradients[self.input_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
