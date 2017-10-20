"""
This code generates a series of neural networks which are trained to perform
arithmetical operations.
The outline is followed from this great overview article:
https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/
with some variations.

The basic process is:
Define a function to learn (addition, subtraction, etc).
Define an operator for that function to be written as a string ('+', '-', etc)
Generate training data:
  + Generate a large number of random operands.
  + Apply the function to the random operands
  + Convert the operands, operator and results to one_hot representation
Train the neural network on the input data.

Some observations:
  + We can create a virtually limitless amount of training data, but there are
    strict bounds around the domain.
  + Is overfitting beneficial? Creating 100k examples of adding two digit numbers
    is likely to create every single example. Is that a problem?
"""