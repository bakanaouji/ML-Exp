import keras


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class WeightHistory(keras.callbacks.Callback):
    def __init__(self, sess, model):
        super().__init__()
        self.sess = sess
        self.model = model
        self.weights = [[] for _ in range(len(self.model.trainable_weights))]

    def on_batch_end(self, batch, logs={}):
        weights = self.sess.run(self.model.trainable_weights)
        for i in range(len(weights)):
            weight = weights[i].flatten()
            self.weights[i].append(weight)
