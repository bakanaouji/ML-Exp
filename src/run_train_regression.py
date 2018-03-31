import argparse

from models.nueral_networks import DenseNetwork
from regression.trainer import Trainer
from utils.loading import load_class


def main():
    parser = argparse.ArgumentParser(description="Train Regression")

    # setting of training
    parser.add_argument('--callbacks', nargs='+', type=str,
                        default=['utils.callbacks.WeightHistory',
                                 'utils.callbacks.LossHistory'])
    parser.add_argument('--target_func', type=str, default='regression.functions.SphereFunction')
    parser.add_argument('--save_path', default='../data', help='Path to save log.')

    args = parser.parse_args()

    # initialize function
    func = load_class(args.target_func)(2)

    # initialize model
    model = DenseNetwork(func.input_dimension(), func.output_dimension(), [64])

    # run train
    trainer = Trainer(args, model, func)
    trainer.train()


if __name__ == '__main__':
    main()
