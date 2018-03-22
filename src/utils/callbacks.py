import keras


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
