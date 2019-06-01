class Optimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, trainables):
        raise NotImplementedError
