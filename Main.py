from Perceptron import Perceptron


def exercise_one():
    perceptron = Perceptron(3)

    targets = [0, 0, 0, 1]

    patterns = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    learning_rate = 0.2

    perceptron.train(patterns, targets, learning_rate, 1000)

    print(perceptron.test([1, 0]))
    print(perceptron.test([0, 1]))
    print(perceptron.test([0, 0]))
    print(perceptron.test([1, 1]))


def exercise_two():
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


if __name__ == "__main__":
    exercise_one()
    #exercise_two()
