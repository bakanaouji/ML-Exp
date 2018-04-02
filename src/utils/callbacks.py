import os
import keras
import pandas as pd


class CallbackBase(keras.callbacks.Callback):
    def __init__(self, sess, model, save_path):
        super().__init__()
        self.sess = sess
        self.model = model
        self.save_path = save_path


class LossHistory(CallbackBase):
    def __init__(self, sess, model, save_path):
        super().__init__(sess, model, save_path)
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append([logs.get('loss'), logs.get('val_loss')])

    def on_train_end(self, logs=None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        loss = pd.DataFrame(self.losses)
        loss.to_csv(self.save_path + '/loss.csv')


class WeightHistory(CallbackBase):
    def __init__(self, sess, model, save_path):
        super().__init__(sess, model, save_path)
        self.sess = sess
        self.model = model
        self.weights = [[] for _ in range(len(self.model.trainable_weights))]

    def on_epoch_end(self, epoch, logs=None):
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
