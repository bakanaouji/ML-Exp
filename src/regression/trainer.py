from utils.data_plotting import plot_scatter_3d
from keras import backend as K

from utils.loading import load_class


class Trainer(object):
    def __init__(self, args, model, func):
        self.save_path = args.save_path
        # initialize parameter of training
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        # initialize model
        self.model = model
        self.model.model.compile(optimizer='rmsprop', loss='mse')
        # initialize function
        self.func = func
        # initialize callbacks
        sess = K.get_session()
        self.callbacks = [load_class(name)(sess, self.model.model,
                                           self.save_path)
                          for name in args.callbacks]

    def train(self):
        x_train, y_train = self.func.sample(self.train_size)
        x_test, y_test = self.func.sample(self.test_size)
        self.model.model.fit(x_train, y_train, batch_size=self.batch_size,
                             epochs=self.epochs, callbacks=self.callbacks,
                             validation_data=(x_test, y_test))
        y_train_predict = self.model.model.predict(x_train)
        y_test_predict = self.model.model.predict(x_test)
        plot_scatter_3d(self.save_path + '/scatter.pdf', x_train[:, 0],
                        x_train[:, 1],
                        y_train)
        plot_scatter_3d(self.save_path + '/train_predict.pdf', x_train[:, 0],
                        x_train[:, 1],
                        y_train_predict)
        plot_scatter_3d(self.save_path + '/test_predict.pdf', x_test[:, 0],
                        x_test[:, 1],
                        y_test_predict)
