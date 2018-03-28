import os
import keras
import pandas as pd


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class WeightHistory(keras.callbacks.Callback):
    def __init__(self, sess, model, save_path):
        super().__init__()
        self.sess = sess
        self.model = model
        self.weights = [[] for _ in range(len(self.model.trainable_weights))]
        self.save_path = save_path

    def on_batch_end(self, batch, logs={}):
        weights = self.sess.run(self.model.trainable_weights)
        for i in range(len(weights)):
            weight = weights[i].flatten()
            self.weights[i].append(weight)

    def on_train_end(self, logs=None):
        for i in range(len(self.weights)):
            save_weight_path = self.save_path + '/layer' + str(i)
            if not os.path.exists(save_weight_path):
                os.makedirs(save_weight_path)
            weight = pd.DataFrame(self.weights[i])
            weight.to_csv(save_weight_path + '/weight.csv')
