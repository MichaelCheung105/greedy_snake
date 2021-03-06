from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization


class Net:
    def __init__(self, shape, learning_rate):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=8, padding='same', activation='relu', input_shape=shape))
        self.model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
        self.model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(4, activation='linear'))
        self.model.compile(optimizer='adam', loss="mae")