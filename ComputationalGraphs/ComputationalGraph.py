import random

from ComputationalGraphs import Commands
from ComputationalGraphs.Commands import Multiplication


# up to ten connections per node
class ComputationalGraph:
    index = 0

    def __init__(self, layers, input_vector, connections):

        assert len(input_vector) % connections == 0
        self.connections = connections
        self.input_vector = input_vector
        self.final = Node()
        self.graph(self.final, layers)

    def simple_graph(self, node):
        for i in range(self.connections):
            node.add_value_channel(self.input_vector[self.index + i])
        self.index += self.connections

    def is_simple(self):
        if type(self.final.input[0]) is int:
            return True
        return False

    def graph(self, node, layers):
        if layers == 2:
            self.simple_graph(node)
        else:
            for i in range(self.connections):
                new_node = Node()
                self.graph(new_node, layers - 1)
                node.add_input_channel(new_node)

    '''
               _  n
           n
        -      -  n
    n             
        -      _  n
           n
               -  n     
    
    '''

    # layer defines number of digits in key -> layer location of node
    # neuron defines the contents of the key -> location of node in layer
    # fill the missing decimal places with zeros

    # neuron from 0 to n - 1
    # layer from 0 to l - 1
    def get_node(self, layer, neuron):
        path = self.get_path(layer, neuron)
        node = self.final
        for turn in list(path):
            node = node.input[int(turn)]
        assert type(node) == Node
        return node

    def derivate(self, layer, neuron):
        path = self.get_path(layer, neuron)
        derivative = 1
        node = self.final
        for turn in list(path):
            derivative *= node.computer.find_derivative(node.compute_uoutput(), int(turn))
            node = node.input[int(turn)]
        return derivative

    def get_path(self, layer, neuron):
        compressed_path = Utils.from_decimal(self.connections, neuron)
        compressed_path = self.fill_path(layer, compressed_path)
        return compressed_path

    @staticmethod
    def fill_path(layer, path):
        return str("0" * (int(layer) - len(str(path)))) + path

    @staticmethod
    def compress_graph(node):
        if node.is_simple():
            return [node, [node.input[i] for i in range(len(node.input))]]
        else:
            return [node, [ComputationalGraph.compress_graph(next_node) for next_node in node.input]]


class Node:

    def __init__(self, computation="multiply"):
        self.input = []
        self.computation = computation
        self.computer = Utils.commands[computation]

    # one of two content modifiers
    def add_input_channel(self, node):
        self.input.append(node)

    # one of two content modifiers
    def add_value_channel(self, value):
        self.input.append(value)

    def compute_output(self):

        if type(self.input[0]) is not int:
            feed_forward = []
            for node in self.input:
                feed_forward.append(node.compute_output())

            return self.computer.compute(feed_forward)
        else:
            return self.computer.compute(self.input)

    def compute_uoutput(self):

        if type(self.input[0]) is int:
            return self.input

        out = []
        for node in self.input:
            out.append(node.compute_output())
        return out

    def is_simple(self):
        if type(self.input[0]) is int:
            return True
        return False

    def __repr__(self):
        return self.computation


class Utils:

    commands = {"multiply": Commands.Multiplication, "add": Commands.Addition}

    # base conversion taken from GeeksForGeeks

    @staticmethod
    # Function to convert a given decimal
    # number to a base 'base' and
    def from_decimal(base, input_num):
        index = 0  # Initialize index of result
        res = ""

        # Convert input number is given base
        # by repeatedly dividing it by base
        # and taking remainder
        while input_num > 0:
            res += Utils.re_val(input_num % base)
            input_num = int(input_num / base)

            # Reverse the result
        res = res[::-1]

        return res

    @staticmethod
    # To return char for a value. For example
    # '2' is returned for 2. 'A' is returned
    # for 10. 'B' for 11
    def re_val(num):

        if 0 <= num <= 9:
            return chr(num + ord('0'))
        else:
            return chr(num - 10 + ord('A'))

            # Utility function to reverse a string

    @staticmethod
    def strev(str):
        len = len(str)
        for i in range(int(len / 2)):
            temp = str[i]
            str[i] = str[len - i - 1]
            str[len - i - 1] = temp


graph = ComputationalGraph(3, [1, 1, 2, 3, 4, 5, 6, 7, 8], 3)
print(ComputationalGraph.compress_graph(graph.final))
node = graph.get_node(1, 2)
print(ComputationalGraph.compress_graph(node))
