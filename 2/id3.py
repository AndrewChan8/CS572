#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node
import math
from node import Leaf, Split

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
  if p == 0 or p == 1:
    return 0
  return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
  if total == 0:
    return 0

  # P(Y=1)
  p_y = py / total
  h_y = entropy(p_y)

  # Conditional entropy H(Y | Xi)
  if pxi == 0 or pxi == total:
    return 0  # No information gain if Xi has no variance

  # Xi = 1
  p_y_given_xi = py_pxi / pxi if pxi != 0 else 0
  h_y_given_xi = entropy(p_y_given_xi)

  # Xi = 0
  py_notxi = py - py_pxi
  p_notxi = total - pxi
  p_y_given_notxi = py_notxi / p_notxi if p_notxi != 0 else 0
  h_y_given_notxi = entropy(p_y_given_notxi)

  # Weighted average
  cond_entropy = (pxi / total) * h_y_given_xi + (p_notxi / total) * h_y_given_notxi

  return h_y - cond_entropy



# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable


# Load data from a file
def read_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  varnames = p.split(header)
  namehash = {}
  for l in f:
    data.append([int(x) for x in p.split(l.strip())])
  return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
  f = open(modelfile, 'w+')
  root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames, used_vars=set(), threshold=0.01):


  # Base case 1: All labels are the same
  labels = [row[-1] for row in data]
  if all(label == labels[0] for label in labels):
    return Leaf(varnames, labels[0])

  # Base case 2: No variables left
  if len(used_vars) == len(varnames) - 1:
    majority = max(set(labels), key=labels.count)
    return Leaf(varnames, majority)

  # Compute best variable using info gain
  best_gain = -1
  best_var = None
  total = len(data)
  py = sum(labels)

  for i in range(len(varnames) - 1):  # skip class label
    if i in used_vars:
        continue

    xi_vals = [row[i] for row in data]
    pxi = sum(xi_vals)
    py_pxi = sum(1 for row in data if row[i] == 1 and row[-1] == 1)
    gain = infogain(py_pxi, pxi, py, total)

    if gain > best_gain:
      best_gain = gain
      best_var = i

  # If no good split found
  if best_gain < threshold or best_var is None:
    majority = max(set(labels), key=labels.count)
    return Leaf(varnames, majority)

  # Partition the data
  data0 = [row for row in data if row[best_var] == 0]
  data1 = [row for row in data if row[best_var] == 1]

  # Recursive build
  used_vars = used_vars.union({best_var})
  left = build_tree(data0, varnames, used_vars, threshold)
  right = build_tree(data1, varnames, used_vars, threshold)

  return Split(varnames, best_var, left, right)

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
  global train
  global varnames
  global test
  global testvarnames
  global root
  (train, varnames) = read_data(trainS)
  (test, testvarnames) = read_data(testS)
  modelfile = modelS

  # build_tree is the main function you'll have to implement, along with
  # any helper functions needed.  It should return the root node of the
  # decision tree.
  root = build_tree(train, varnames)
  print_model(root, modelfile)


def runTest():
  correct = 0
  # The position of the class label is the last element in the list.
  yi = len(test[0]) - 1
  for x in test:
    # Classification is done recursively by the node class.
    # This should work as-is.
    pred = root.classify(x)
    if pred == x[yi]:
      correct += 1
  acc = float(correct) / len(test)
  return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
  if (len(argv) != 3):
    print('Usage: python3 id3.py <train> <test> <model>')
    sys.exit(2)
  loadAndTrain(argv[0], argv[1], argv[2])

  acc = runTest()
  print("Accuracy: ", acc)


if __name__ == "__main__":
  main(sys.argv[1:])
