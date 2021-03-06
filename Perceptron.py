import random


class Perceptron:

    def __init__(self, num_weights):
        self.weights = [random.randrange(-1, 1) for i in range(num_weights)]

    def sum(self, inputs):
        sum = self.weights[0]

        for i in range(len(self.weights)-1):
            sum += inputs[i] * self.weights[i+1]

        return sum

    def test(self, inputs):

        if self.sum(inputs) >= 0:
            return 1

        return 0

    def train(self, patterns, targets, learning_rate, epochs=10000):
        has_error = True
        iterations = 0
        while has_error and iterations < epochs:
            has_error = False
            for i, (inputs, target) in enumerate(zip(patterns, targets)):
                error = target - self.test(inputs)

                if error != 0:
                    has_error = True

                self.weights[0] = self.weights[0] + learning_rate * error
                for j in range(len(self.weights)-1):
                    self.weights[j+1] = self.weights[j+1] + learning_rate * error * inputs[j]

            iterations += 1

