from optimizers.optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, learning_rate):
        Optimizer.__init__(self, learning_rate)

    def update(self, training_nodes):
        for node in training_nodes:
            partial = node.gradients[node]
            node.output -= self.learning_rate * partial
