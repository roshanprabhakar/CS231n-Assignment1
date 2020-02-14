from abc import abstractmethod


class Command:

    id = "COMMAND"

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def compute(numer_list):
        pass

    @staticmethod
    @abstractmethod
    def find_derivative(numer_list, index):
        pass


class Multiplication(Command):

    @staticmethod
    def compute(numer_list):
        out = 1
        for num in numer_list:
            out *= num
        return out

    @staticmethod
    def find_derivative(numer_list, index):
        out = 1
        for i in range(len(numer_list)):
            if i != index:
                out *= numer_list[i]
        return out


class Addition(Command):

    @staticmethod
    def compute(numer_list):
        out = 0
        for num in numer_list:
            out += num
        return out

    def find_derivative(numer_list, index):
        return 1