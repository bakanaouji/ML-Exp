import argparse

from models.nueral_networks import DenseNetwork
from regression.functions import SphereFunction
from regression.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Regression")

    args = parser.parse_args()

    # initialize model
    model = DenseNetwork(2, 1, [16, 8])

    # initialize function
    func = SphereFunction(2)

    trainer = Trainer(args, model, func)
    trainer.train()


if __name__ == '__main__':
    main()
