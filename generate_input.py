#!/usr/bin/env python3

from PIL import Image
from keras.models import load_model
from utils import SIZE, generate_patches, get_horizontal_predicts, get_vertical_predicts

import argparse
import glob
import os


def generate_input(image, model, delta, patch):
    print('Generating horizontal predicts...')
    hors = get_horizontal_predicts(image, model, delta, patch)
    print('Generating vertical predicts...')
    vers = get_horizontal_predicts(image, model, delta, patch)
    print('Done')
    return hors, hors


def print_1d_array(array, file):
    print(' '.join(map(lambda x: '{:.6}'.format(x), array)), file=file)


def print_2d_array(array, file):
    for i in range(len(array)):
        print_1d_array(array[i], file=file)


def load_answers(answers_path):
    answers = {}
    with open(answers_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) % 2 == 0
        for i in range(0, len(lines), 2):
            name = lines[i].strip()
            permutation = list(map(int, lines[i + 1].split()))
            answers[name] = permutation
    return answers


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
    parser.add_argument('--data',
                        type=str,
                        default='data/data_train/64',
                        help='Path to the train data')
    parser.add_argument('--answers',
                        type=str,
                        default=None,
                        help='Path to the train data answers')
    parser.add_argument('--model',
                        type=str,
                        default='model.h5',
                        help='Path to the model')
    args = parser.parse_args()

    model = load_model(args.model)
    if args.answers:
        answers = load_answers(answers_path=args.answers)
    else:
        answers = {}

    for file in glob.glob(os.path.join(args.data, '*.png')):
        print('Processing {}...'.format(file))
        name = os.path.basename(file)
        
        image = Image.open(file).convert('L')
        hors, vers = generate_input(image=image,
                                    model=model,
                                    delta=args.delta,
                                    patch=args.patch)

        with open(os.path.join(os.path.dirname(file), name + '.txt'), 'w') as f:
            print(SIZE // args.patch, file=f)
            print_2d_array(hors, file=f)
            print_2d_array(vers, file=f)
            if answers:
                print(' '.join(map(str, answers[name])), file=f)
