from abc import abstractmethod


class Command:

    id = "COMMAND"

    def __init__(self):
        pass

    @abstractmethod
    def compute(self, numer_list):
        pass


class Multiplication(Command):

    def compute(self, numer_list):
        out = 1
        for num in numer_list:
            out *= num
        return out


class Addition(Command):

    def compute(self, numer_list):
        out = 0
        for num in numer_list:
            out += num
        return out