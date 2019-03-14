from Perceptron import Perceptron

if __name__ == "__main__":

    perceptron = Perceptron(26)

    D = [0, 1]

    P = [
        [
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
        ],

    ]

    learning_rate = 0.2

    perceptron.train(P, D, learning_rate, 1000)

    print(perceptron.test([
            1, 1, 1, 1, 1,
            1, 0, 1, 0, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ]))

    print(perceptron.test([
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
        ]))

    print(perceptron.test([
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ]))

    print(perceptron.test([
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
        ]))