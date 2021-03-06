from ops.node import Node
import numpy as np


class Linear(Node):

    def __init__(self, input, weights, bias):
        Node.__init__(self, [input, weights, bias])

    def compute(self):
        """
        Compute dot product plus bias
        """
        X = self.input_nodes[0].output
        W = self.input_nodes[1].output
        b = self.input_nodes[2].output
        self.output = np.dot(X, W) + b

    def backpass(self):
        """
        Backpropagate gradients from output nodes towards input nodes.
        """

        # clear gradients of input nodes
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        # update gradients of input nodes based on gradients from output nodes
        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += np.dot(grad_cost, self.input_nodes[1].output.T)
            self.gradients[self.input_nodes[1]] += np.dot(self.input_nodes[0].output.T, grad_cost)
            self.gradients[self.input_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)
