#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)

# Train a logistic regression model using batch gradient descent
def sigmoid(z):
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = exp(z)
        return ez / (1.0 + ez)

# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    for iteration in range(MAX_ITERS):
        gradW = [0.0] * numvars
        gradB = 0.0

        for x, y in data:
            if y == -1:
                y = 0

            z = sum(w[i] * x[i] for i in range(numvars)) + b
            yHat = sigmoid(z)
            error = yHat - y 

            for i in range(numvars):
                gradW[i] += error * x[i]
            gradB += error

        for i in range(numvars):
            gradW[i] += l2_reg_weight * w[i]

        gradMagnitude = sqrt(sum(g ** 2 for g in gradW) + gradB ** 2)
        if gradMagnitude < 1e-4:
            break

        for i in range(numvars):
            w[i] -= eta * gradW[i]
        b -= eta * gradB

    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    z = sum(w[i] * x[i] for i in range(len(w))) + b
    return sigmoid(z)

# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
