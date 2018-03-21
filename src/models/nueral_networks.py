from keras.layers import Dense

from keras.models import Sequential


class DenseNetwork(object):
    def __init__(self, n, output_dim, hidden_sizes):
        self.model = Sequential()
        self.model.add(Dense(hidden_sizes[0], activation='relu', input_dim=n))
        for i in range(1, len(hidden_sizes)):
            self.model.add(Dense(hidden_sizes[i], activation='relu'))
        self.model.add(Dense(output_dim))
