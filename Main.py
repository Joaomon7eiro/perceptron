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

    targets = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    patterns = [
        [
            1, 1, 1, 1, 1,
            1, 0, 1, 0, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            0, 1, 1, 1, 0,
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
        [
            1, 1, 1, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 0, 1, 1,
        ],
        [
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 1, 0, 1,
            1, 1, 0, 1, 1,
            1, 0, 0, 0, 1,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
        ]

    ]

    learning_rate = 0.2

    perceptron.train(patterns, targets, learning_rate, 1000)

    error = False

    result = perceptron.test([
        1, 1, 1, 1, 1,
        1, 0, 1, 0, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    ])

    if result != 0:
        error = True

    print(result)

    result = perceptron.test([
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    ])

    if result != 0:
        error = True

    print(result)

    result = perceptron.test([
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
    ])
    if result != 1:
        error = True

    print(result)

    result = perceptron.test([
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
    ])
    if result != 1:
        error = True

    print(result)

    return error

def exercise_three():
    #           a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t  u  v  w  x  y  z
    targets1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    targets2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    targets3 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    targets4 = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    targets5 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    patterns = [
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
        ],
        [
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 0,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
        ],
        [
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 0,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 0, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
        ],
        [
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
        ],
        [
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            1, 0, 1, 0, 0,
            1, 0, 1, 0, 0,
            1, 1, 1, 0, 0,
        ],
        [
            1, 0, 0, 0, 1,
            1, 0, 0, 1, 0,
            1, 1, 1, 0, 0,
            1, 0, 0, 1, 0,
            1, 0, 0, 0, 1,
        ],
        [
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
        ],
        [
            1, 0, 0, 0, 1,
            1, 1, 0, 1, 1,
            1, 0, 1, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
        ],
        [
            1, 0, 0, 0, 1,
            1, 1, 0, 0, 1,
            1, 0, 1, 0, 1,
            1, 0, 0, 1, 1,
            1, 0, 0, 0, 1,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
        ],
        [
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 1, 0, 1,
            1, 0, 0, 1, 1,
            0, 1, 1, 1, 1,
        ],
        [
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 0,
            1, 0, 0, 1, 0,
            1, 0, 0, 0, 1,
        ],
        [
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
        ],
        [
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
        ],
        [
            1, 0, 0, 0, 1,
            0, 0, 0, 0, 0,
            0, 1, 0, 1, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 1, 0, 1,
            1, 1, 0, 1, 1,
            1, 0, 0, 0, 1,
        ],
        [
            1, 0, 0, 0, 1,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            1, 0, 0, 0, 1,
        ],
        [
            1, 0, 0, 0, 1,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
        ],
        [
            1, 1, 1, 1, 1,
            0, 0, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 0, 0,
            1, 1, 1, 1, 1,
        ]
    ]

    perceptron1 = Perceptron(26)
    perceptron2 = Perceptron(26)
    perceptron3 = Perceptron(26)
    perceptron4 = Perceptron(26)
    perceptron5 = Perceptron(26)

    learning_rate = 0.2

    perceptron1.train(patterns, targets1, learning_rate, 1000)
    perceptron2.train(patterns, targets2, learning_rate, 1000)
    perceptron3.train(patterns, targets3, learning_rate, 1000)
    perceptron4.train(patterns, targets4, learning_rate, 1000)
    perceptron5.train(patterns, targets5, learning_rate, 1000)

    error = False


    result = perceptron1.test([
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    ])

    if result != 1:
        error = True


    print(result)

    result = perceptron2.test([
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    ])


    if result != 0:
        error = True

    print(result)

    result = perceptron3.test([
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    ])


    if result != 0:
        error = True

    print(result)

    result = perceptron4.test([
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    ])

    if result != 1:
        error = True

    print(result)

    result = perceptron5.test([
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    ])


    if result != 1:
        error = True

    print(result)

    return error



def exercise_four():
    perceptron_xor = Perceptron(3)

    target_xor = [0, 1, 1, 0]

    patterns = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    learning_rate = 0.2

    perceptron_xor.train(patterns, target_xor, learning_rate)

    print(perceptron_xor.test([0, 0]))
    print(perceptron_xor.test([0, 1]))
    print(perceptron_xor.test([1, 0]))
    print(perceptron_xor.test([1, 1]))

def exercise_five():
    perceptron_nand = Perceptron(3)
    perceptron_or = Perceptron(3)
    perceptron_and = Perceptron(3)

    target_nand = [1, 1, 1, 0]
    target_or = [0, 1, 1, 1]
    target_and = [0, 0, 0, 1]

    patterns = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    learning_rate = 0.2

    perceptron_nand.train(patterns, target_nand, learning_rate)
    perceptron_or.train(patterns, target_or, learning_rate)
    perceptron_and.train(patterns, target_and, learning_rate)

    and_inputs = []

    and_inputs.append(perceptron_nand.test([0, 0]))
    and_inputs.append(perceptron_or.test([0, 0]))
    print(perceptron_and.test(and_inputs))

    and_inputs = []

    and_inputs.append(perceptron_nand.test([0, 1]))
    and_inputs.append(perceptron_or.test([0, 1]))
    print(perceptron_and.test(and_inputs))

    and_inputs = []

    and_inputs.append(perceptron_nand.test([1, 0]))
    and_inputs.append(perceptron_or.test([1, 0]))
    print(perceptron_and.test(and_inputs))

    and_inputs = []

    and_inputs.append(perceptron_nand.test([1, 1]))
    and_inputs.append(perceptron_or.test([1, 1]))
    print(perceptron_and.test(and_inputs))

def test_exercise_two():
    count_error = 0
    for i in range(10):
        if exercise_two():
            count_error += 1

    print("Taxa de erro", count_error * 10, "%")

def test_exercise_three():
    count_error = 0
    for i in range(10):
        if exercise_three():
            count_error += 1

    print("Taxa de erro", count_error * 10, "%")

if __name__ == "__main__":
    print("Exercicio 1")
    exercise_one()

    print("Exercicio 2")
    exercise_two()

    print("Exercicio 3")
    exercise_three()

    print("Exercicio 4")
    exercise_four()

    print("Exercicio 5")
    exercise_five()

    print("Teste Exercicio 2")
    test_exercise_two()

    print("Teste Exercicio 3")
    test_exercise_three()
