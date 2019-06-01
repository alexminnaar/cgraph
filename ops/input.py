from ops.node import Node


class Input(Node):

    def __init__(self):
        Node.__init__(self)


    def compute(self,output=None):
        """
        Input nodes don't need to compute anything, they just hold the input values.
        """
        pass

    def backpass(self):

        self.gradients = {self:0}

        for node in self.output_nodes:
            self.gradients[self]+= node.gradients[self]


