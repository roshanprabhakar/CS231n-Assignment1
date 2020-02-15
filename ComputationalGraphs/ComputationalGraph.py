import random

from ComputationalGraphs import Commands
from ComputationalGraphs.Commands import Multiplication


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

    def graph(self, node, layers):
        if layers == 2:
            self.simple_graph(node)
        else:
            for i in range(self.connections):
                new_node = Node()
                self.graph(new_node, layers - 1)
                node.add_input_channel(new_node)

    def is_simple(self):
        if type(self.final.input[0]) is int:
            return True
        return False

    @staticmethod
    def compress_graph(node):
        if node.is_simple():
            return [node, [node.input[i] for i in range(len(node.input))]]
        else:
            return [node, [ComputationalGraph.compress_graph(next_node) for next_node in node.input]]



class Node:

    def __init__(self, computation="add"):
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

    def is_simple(self):
        if type(self.input[0]) is int:
            return True
        return False

    def __repr__(self):
        return self.computation


class Utils:
    commands = {"multiply": Commands.Multiplication, "add": Commands.Addition}


graph = ComputationalGraph(3, [0, 1, 2, 3, 4, 5, 6, 7, 8], 3)
compressed = ComputationalGraph.compress_graph(graph.final)
# print(graph.final.compute_output())
