from Node import Node
import numpy as np


class MSE(Node):

    def __init__(self, y, y_hat):
        Node.__init__(self, [y, y_hat])

    def compute(self):
        y = self.input_nodes[0].output.reshape(-1, 1)
        y_hat = self.input_nodes[1].output.reshape(-1, 1)
        self.y_len = self.input_nodes[0].output.shape[0]
        self.y_diff = y - y_hat

        self.output = np.mean(self.y_diff ** 2)

    def backpass(self):
        self.gradients[self.input_nodes[0]] = (2 / self.y_len) * self.y_diff
        self.gradients[self.input_nodes[1]] = (-2 / self.y_len) * self.y_diff
