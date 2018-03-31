from utils.data_plotting import plot_scatter_3d
from keras import backend as K

from utils.loading import load_class


class Trainer(object):
    def __init__(self, args, model, func):
        self.save_path = args.save_path
        # initialize model
        self.model = model
        self.model.model.compile(optimizer='rmsprop', loss='mse')
        # initialize function
        self.func = func
        # initialize callbacks
        sess = K.get_session()
        self.callbacks = [load_class(name)(sess, self.model.model, self.save_path) for name in args.callbacks]

    def train(self):
        x_data, y_data = self.func.sample(10000)
        self.model.model.fit(x_data, y_data, batch_size=16, epochs=10,
                             callbacks=self.callbacks)
        y_predict = self.model.model.predict(x_data)
        plot_scatter_3d(self.save_path + '/scatter.pdf', x_data[:, 0], x_data[:, 1],
                        y_data)
        plot_scatter_3d(self.save_path + '/predict.pdf', x_data[:, 0], x_data[:, 1],
                        y_predict)
