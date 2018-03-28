from utils.callbacks import WeightHistory
from utils.data_plotting import plot_scatter_3d
from keras import backend as K


class Trainer(object):
    def __init__(self, args, model, func):
        # initialize model
        self.model = model
        self.model.model.compile(optimizer='rmsprop', loss='mse')
        # initialize function
        self.func = func

    def train(self):
        sess = K.get_session()
        callbacks = [WeightHistory(sess, self.model.model, '../data')]
        x_data, y_data = self.func.sample(10000)
        self.model.model.fit(x_data, y_data, batch_size=16, epochs=10,
                             callbacks=callbacks)
        y_predict = self.model.model.predict(x_data)
        plot_scatter_3d('../data/scatter.pdf', x_data[:, 0], x_data[:, 1],
                        y_data)
        plot_scatter_3d('../data/predict.pdf', x_data[:, 0], x_data[:, 1],
                        y_predict)
