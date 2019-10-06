#!/usr/bin/env python3

from PIL import Image
from keras.models import load_model
from utils import SIZE, get_horizontal_scores, get_vertical_scores

import argparse


def generate_input(image, model, delta, patch):
    return get_horizontal_scores(image, model, delta, patch), get_vertical_scores(image, model, delta, patch)


def print_2d_array(array):
    for i in range(len(array)):
        print(' '.join(map(lambda x: '{:.6}'.format(x), array[i])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patch',
                        type=int,
                        default=64,
                        help='Size of the image patch')
    parser.add_argument('--delta',
                        type=int,
                        default=8,
                        help='Size of the image delta')
    parser.add_argument('--image',
                        type=str,
                        default='data/data_train/16-sources/0010.png',
                        help='Path to the image')
    parser.add_argument('--model',
                        type=str,
                        default='model.h5',
                        help='Path to the model')
    args = parser.parse_args()

    image = Image.open(args.image).convert('L')
    model = load_model(args.model)

    hors, vers = generate_input(image=image,
                                model=model,
                                delta=args.delta,
                                patch=args.patch)

    print(SIZE // args.patch)
    print_2d_array(hors)
    print_2d_array(vers)
