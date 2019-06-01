from ops.node import Node
from sklearn.metrics import log_loss


class BinaryCrossEntropy(Node):

    def __init__(self, y, y_hat):
        Node.__init__(self, [y, y_hat])

    def compute(self):
        """
        Computing the binary cross-entropy loss
        """
        y = self.input_nodes[0].output.reshape(-1, 1)
        y_hat = self.input_nodes[1].output.reshape(-1, 1)
        self.y_len = self.input_nodes[0].output.shape[0]
        self.y_diff = y - y_hat
        self.output = log_loss(y, y_hat)

    def backpass(self):
        """
        backpropagate the gradient to the input nodes
        :return:
        """
        self.gradients[self.input_nodes[0]] = self.y_diff
        self.gradients[self.input_nodes[1]] = self.y_diff
