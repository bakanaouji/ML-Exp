import argparse
import numpy as np
import tensorflow as tf

from models.nueral_networks import DenseNetwork
from regression.trainer import Trainer
from utils.loading import load_class


def main():
    parser = argparse.ArgumentParser(description="Train Regression")

    # setting of training
    parser.add_argument('--callbacks', nargs='+', type=str,
                        default=['utils.callbacks.WeightHistory',
                                 'utils.callbacks.LossHistory'])
    parser.add_argument('--target_func', type=str,
                        default='regression.functions.SphereFunction')
    parser.add_argument('--save_path', default='../data',
                        help='Path to save log.')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed.')

    args = parser.parse_args()

    # set random seed
    if args.seed >= 0:
        np.random.seed(args.seed)
        tf.set_random_seed(np.random.randint(2 ** 32))

    # initialize function
    func = load_class(args.target_func)(2)

    # initialize model
    model = DenseNetwork(func.input_dimension(), func.output_dimension(), [64])

    # run train
    trainer = Trainer(args, model, func)
    trainer.train()


if __name__ == '__main__':
    main()
