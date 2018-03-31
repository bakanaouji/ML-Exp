import argparse

from models.nueral_networks import DenseNetwork
from regression.functions import SphereFunction
from regression.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Regression")

    # setting of training
    parser.add_argument('--callbacks', nargs='+', type=str,
                        default=['utils.callbacks.WeightHistory',
                                 'utils.callbacks.LossHistory'])

    args = parser.parse_args()

    # initialize function
    func = SphereFunction(2)

    # initialize model
    model = DenseNetwork(func.input_dimension(), func.output_dimension(), [64])

    # run train
    trainer = Trainer(args, model, func)
    trainer.train()


if __name__ == '__main__':
    main()
