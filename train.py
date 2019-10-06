#!/usr/bin/env python3

from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import argparse
import glob
import numpy as np
import os
import random
import sys


SIZE = 512


def all_transpositions(im):
    yield im
    yield im.transpose(Image.FLIP_LEFT_RIGHT)
    yield im.transpose(Image.FLIP_TOP_BOTTOM)
    yield im.transpose(Image.ROTATE_180)


def random_patch(im, delta, patch):
    i = random.randrange(0, SIZE, delta)
    j = random.randrange(0, SIZE, patch)
    return im.crop((i, j, i + delta, j + patch))


def generate_patches(im, delta, patch):
    total = 0
    for i in range(0, SIZE, 2 * patch):
        for j in range(0, SIZE, patch):
            p = im.crop((i + patch - delta, j, i + patch + delta, j + patch))
            for image in all_transpositions(p):
                yield image, 1
                total += 1
    for i in range(0, SIZE, patch):
        for j in range(0, SIZE, 2 * patch):
            p = im.crop((i, j + patch - delta, i + patch, j + patch + delta))
            for image in all_transpositions(p.transpose(Image.ROTATE_90)):
                yield image, 1
                total += 1
    for _ in range(total):
        a = random_patch(im, delta, patch)
        b = random_patch(im, delta, patch)
        image = Image.new('L', (2 * delta, patch))
        image.paste(a, (0, 0))
        image.paste(b, (delta, 0))
        yield image, 0


def generate_data(data_dir, delta, patch):
    xs, ys = [], []
    for path in glob.glob(os.path.join(data_dir, '*.png')):
        im = Image.open(path).convert('L')
        assert im.height == SIZE
        assert im.width == SIZE
        for x, y in generate_patches(im, delta, patch):
            if random.random() < 0.25:
                xs.append(np.asarray(x) / 255)
                ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    xs = np.expand_dims(xs, axis=-1)
    return xs, ys


def create_model(delta, patch):
    input_shape = (patch, 2 * delta, 1)
    drop_rate = 0.25

    model = Sequential()

    model.add(Conv2D(input_shape=input_shape,
                     filters=64,
                     kernel_size=(5, 5),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))

    model.add(Conv2D(input_shape=input_shape,
                    filters=64,
                    kernel_size=(5, 5),
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return model


def go(data_dir, model_path, delta, patch):
    Xs, ys = generate_data(data_dir, delta, patch)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys,
                                                            test_size=0.2,
                                                            random_state=42)
    model = create_model(delta, patch)

    callbacks = [
        ModelCheckpoint(model_path,
                        monitor='val_binary_accuracy', 
                        save_best_only=True, 
                        mode='max',
                        verbose=1)
    ]
    model.fit(Xs_train, ys_train,
              batch_size=64,
              epochs=10,
              validation_data=(Xs_test, ys_test),
              callbacks=callbacks,
              shuffle=True,
              verbose=1)


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
                        default='.',
                        help='Path to the directory with training images')
    parser.add_argument('--model',
                        type=str,
                        default='model.h5',
                        help='Path for the model to save')
    args = parser.parse_args()

    go(data_dir=args.data,
       model_path=args.model,
       delta=args.delta,
       patch=args.patch)
