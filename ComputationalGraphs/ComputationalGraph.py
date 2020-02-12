from ComputationalGraphs import Commands


class ComputationalGraph:

    def __init__(self):

        # lat layer, 1 neuron
        n2p0 = Node()

        # second last layer, 2 neuron
        n1p0 = Node()
        n1p1 = Node()

        # input layer
        n0p0 = 4
        n0p1 = 5
        n0p2 = 3
        n0p3 = 7

        n2p0.add_input_channel(n1p0)
        n2p0.add_input_channel(n1p1)

        n1p0.add_input_value(n0p0)
        n1p0.add_input_value(n0p1)

        n1p1.add_input_value(n0p2)
        n1p1.add_input_value(n0p3)

        print(n2p0.compute_output())


class Node:

    def __init__(self, computation="multiply"):
        self.input = []
        self.computer = Utils.commands[computation]

    # one of two content modifiers
    def add_input_channel(self, node):
        self.input.append(node)

    # one of two content modifiers
    def add_input_value(self, value):
        self.input.append(value)

    def compute_output(self):

        if type(self.input[0]) is int:
            return self.computer.compute(self.input)
        else:
            feed_forward = []
            for node in self.input:
                feed_forward.append(node.computeOutput())

            return self.computer.compute(self.input)


class Utils:
    commands = {"multiply": Commands.Multiplication, "add": Commands.Addition}


graph = ComputationalGraph()