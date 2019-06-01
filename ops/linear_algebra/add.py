from ops.node import Node

class Add(Node):

    def __init__(self, operand_1,operand_2):
        Node.__init__(self,[operand_1,operand_2])

    def compute(self):

        operand_1 = self.input_nodes[0].output
        operand_2 = self.input_nodes[1].output

        self.output = operand_1 + operand_2