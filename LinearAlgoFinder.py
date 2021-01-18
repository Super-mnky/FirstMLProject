import tensorflow as tf
# an open source library for machine learning! Has a focus on traning and inference of deep neural networks.

import numpy as np
# an open source library for dealing with large multi-dimentional arrays and matricies! + the math functions to boot

from tensorflow import keras
# an open source library that provides inference for nerual networks.

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# I've defined a very simple neural network. 
# The network has one layer, with one neuron, and the input shape is to only 1 value.

model.compile(optimizer='sgd', loss='mean_squared_error')
# optimizer & loss = define function that tries to minimize the loss (how far off the guess was from the prediction), and optimize the next guess

# Two arrays have been provided as data. The relationship I'm trying to train this model to produce is 'y = 3x + 1.'
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# The command to train the neural network over 'epochs'. 
model.fit(xs, ys, epochs=250)

# With a trained network, we can ask it to make a prediction! Let's give it an x, and see what pops out.
print(model.predict([10.0]))