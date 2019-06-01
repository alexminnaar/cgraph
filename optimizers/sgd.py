from optimizers.optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, learning_rate):
        Optimizer.__init__(self, learning_rate)

    def update(self, training_nodes):
        """
        For nodes with trainable weights, update the weights based on previously computed gradients.
        :param training_nodes: Nodes with trainable weights.
        """
        for node in training_nodes:
            partial = node.gradients[node]
            node.output -= self.learning_rate * partial
