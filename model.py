import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np
import random

class Net:
    def __init__(self, shape, epsilon):
        self.epsilon = epsilon

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=shape))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(5, activation='softmax'))
        self.model.compile(optimizer='adam', loss="mse")

    def suggest(self, state, action_space):
        input = np.expand_dims(state, axis=0)
        q_values = self.model.predict(input)

        if np.random.rand() < self.epsilon:
            action = random.choice(action_space)
        else:
            action = action_space[np.argmax(q_values)]
        return action

