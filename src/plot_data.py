import argparse

from utils.data_plotting import plot_data


def main():
    parser = argparse.ArgumentParser(description='Plot data')

    # setting
    parser.add_argument('--x_min', type=float, default=0.0,
                        help='Min value of x axis')
    parser.add_argument('--x_max', type=float, default=6500.0,
                        help='Max value of x axis')
    parser.add_argument('--y_min', type=float, default=-2.0,
                        help='Min value of y axis')
    parser.add_argument('--y_max', type=float, default=2.0,
                        help='Max value of y axis')
    parser.add_argument('--save_path',
                        default='../data/layer0/weight.csv')

    args = parser.parse_args()

    # plot data
    print('----------Plot Data----------')
    plot_data(args.save_path, args.x_min, args.x_max, args.y_min, args.y_max)


if __name__ == '__main__':
    main()
