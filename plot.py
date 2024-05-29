import argparse
from argparse import BooleanOptionalAction

from rlm.utils import plot_results, save_plot_as_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True, type=str)
    parser.add_argument('--x_axis', default='steps', type=str)
    parser.add_argument('--smoothing', default=None, type=int)
    parser.add_argument('--save_pdf', default=False, action=BooleanOptionalAction)
    args = parser.parse_args()

    if args.save_pdf:
        save_plot_as_pdf(
            args.load,
            x_axis=args.x_axis,
            smoothing=args.smoothing)
    else:
        plot_results(
            args.load,
            x_axis=args.x_axis,
            smoothing=args.smoothing)

if __name__ == '__main__':
    main()
