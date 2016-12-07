# eqnn
Neural network channel equalizer for wireless signals to prove that LMS is equivalent to a Neural Network in very special conditions.
Desired feature list after main goal is accomplished:
-Reimplement ANN as a convolutional network in order to pipeline data processing
-Actually implement a full TF implementation of an equalizer with RELU activation functions
-->Look into RNN as a means to keep track of channel "history" 
-Experiment with a neural net turbo decoder?


# Version Log

1.0
--------
Added testing framework including:
-RLS and LMS as baselines
-Naive Tensorflow implementation of an equalizer
