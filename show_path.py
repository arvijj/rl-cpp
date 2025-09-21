import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_dir', default=None, type=str)
    parser.add_argument('--type', default='eval', type=str)
    parser.add_argument('--episode', default=1, type=int)
    parser.add_argument('--steps', default=None, type=int)
    parser.add_argument('--meters_per_pixel', default=0.0375, type=float)
    args = parser.parse_args()

    # Read data
    filename = args.type + f'_metrics_{args.episode}.csv'
    f = open(os.path.join(args.metrics_dir, filename), 'r')
    names = f.readline().replace('\n', '').split(',')
    metrics = {name: [] for name in names}
    for line in f:
        values = line.replace('\n', '').split(',')
        for name, value in zip(names, values):
            metrics[name].append(None if value == '' else float(value))

    # Print T90/T99
    coverage = np.array(metrics['coverage'])
    gc_idx_90 = np.argmax(coverage >= 0.9)
    gc_idx_99 = np.argmax(coverage >= 0.99)
    print('T90:', round(metrics['time'][gc_idx_90] / 60, 2))
    print('T99:', round(metrics['time'][gc_idx_99] / 60, 2))

    # Print path length
    print('Length:', round(np.array(metrics['length'])[-1], 2))

    # Print accumulated turns
    print('Turns:', round(np.array(metrics['turns'])[-1], 2))

    # Print collisions
    collisions = np.array(metrics['collisions'])
    print('Coll (all):', int(collisions[-1]))
    # only count first collision out of a sequence
    collisions = np.diff(collisions)
    print('Coll (diff):', np.sum(np.diff(collisions) > 0))

    # Keep specified number of steps for plotting
    if args.steps is not None:
        for name in names:
            metrics[name] = metrics[name][:args.steps]

    # Read map
    filename = args.type + f'_map_{args.episode}.png'
    img = cv2.imread(os.path.join(args.metrics_dir, filename), flags=cv2.IMREAD_GRAYSCALE)
    img = np.fliplr(img.transpose(1, 0))
    size_p, _ = img.shape
    obstacle_map = np.zeros_like(img, dtype=float)
    obstacle_map[img == 0] = 1

    # Draw map
    fig, axes = plt.subplot_mosaic('A', constrained_layout=True)
    fig.set_size_inches(8, 8)
    for ax in axes:
        axes[ax].get_xaxis().set_visible(False)
        axes[ax].get_yaxis().set_visible(False)
    image = np.ones(img.shape + (3,), dtype=float)
    image[obstacle_map > 0] = 0
    axes['A'].imshow(np.flip(image.transpose(1, 0, 2), axis=0), interpolation='nearest')

    # Draw path
    xs = np.array(metrics['x']) / args.meters_per_pixel
    ys = np.array(metrics['y']) / args.meters_per_pixel
    axes['A'].plot(xs - 0.5, size_p - ys - 0.5, '-', color='black')
    axes['A'].plot(xs[0] - 0.5, size_p - ys[0] - 0.5, '^', color='red', markersize=15)
    axes['A'].plot(xs[-1] - 0.5, size_p - ys[-1] - 0.5, 's', color='green', markersize=15)
    axes['A'].set_xlim([-0.5, size_p - 0.5])
    axes['A'].set_ylim([size_p - 0.5, -0.5])

    plt.show()

if __name__ == '__main__':
    main()
