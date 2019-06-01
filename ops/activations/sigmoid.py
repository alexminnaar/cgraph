from ops.node import Node
import numpy as np


class Sigmoid(Node):

    def __init__(self, node):
        Node.__init__(self, [node])

    def compute(self):
        input_value = self.input_nodes[0].output
        self.output = 1. / (1. + np.exp(-input_value))

    def backpass(self):
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        for node in self.output_nodes:
            grad_cost = node.gradients[self]
            sigmoid = self.output
            self.gradients[self.input_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
