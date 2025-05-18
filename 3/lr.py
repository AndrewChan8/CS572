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
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    for it in range(MAX_ITERS):
        dw = [0.0] * numvars
        db = 0.0
        total_loss = 0.0

        for (x, y) in data:
            z = sum(w[i] * x[i] for i in range(numvars)) + b
            yz = y * z
            sig = 1.0 / (1.0 + exp(-yz))

            # Gradient components
            factor = (sig - 1) * y
            for i in range(numvars):
                dw[i] += factor * x[i]
            db += factor

            # Logistic loss (with log-sum-exp trick)
            total_loss += log(1 + exp(-yz))

        # L2 regularization gradient and loss
        for i in range(numvars):
            dw[i] = dw[i] / len(data) + l2_reg_weight * w[i]
            total_loss += 0.5 * l2_reg_weight * w[i] ** 2
        db = db / len(data)

        grad_norm = sqrt(sum(g ** 2 for g in dw) + db ** 2)

        if grad_norm < 0.0001:
            print("Converged.")
            break

        for i in range(numvars):
            w[i] -= eta * dw[i]
        b -= eta * db

    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    #
    # YOUR CODE HERE
    #
    z = sum(w[i] * x[i] for i in range(len(x))) + b
    prob = 1.0 / (1.0 + exp(-z))
    return prob

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
