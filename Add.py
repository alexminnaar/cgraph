from Node import Node

class Add(Node):

    def __init__(self, operand_1,operand_2):
        Node.__init__(self,[operand_1,operand_2])

    def compute(self):

        operand_1_value = self.input_nodes[0].output
        operand_2_value = self.input_nodes[1].output

        self.output = operand_1_value + operand_2_value