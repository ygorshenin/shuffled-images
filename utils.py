#!/usr/bin/env python3

from PIL import Image
from keras.models import load_model

import numpy as np


SIZE = 512


def generate_patches(image, patch):
    for y in range(0, SIZE, patch):
        for x in range(0, SIZE, patch):
            yield image.crop((x, y, x + patch, y + patch))


def combine_horizontal(left, right, delta, patch):
    l = left.crop((patch - delta, 0, patch, patch))
    r = right.crop((0, 0, delta, patch))
    im = Image.new('L', (2 * delta, patch))
    im.paste(l, (0, 0))
    im.paste(r, (delta, 0))
    return np.asarray(im) / 255


def combine_vertical(upper, lower, delta, patch):
    return combine_horizontal(left=upper.transpose(Image.ROTATE_90),
                              right=lower.transpose(Image.ROTATE_90),
                              delta=delta,
                              patch=patch)


def get_predicts(image, model, delta, patch, combine):
    patches = list(generate_patches(image=image, patch=patch))
    hors = []
    for i in range(len(patches)):
        for j in range(len(patches)):
            hors.append(combine(patches[i], patches[j], delta, patch))
    hors = np.array(hors)
    hors = np.expand_dims(hors, axis=-1)

    scores = model.predict(hors)
    return scores.reshape((len(patches), len(patches)))


def get_horizontal_predicts(image, model, delta, patch):
    return get_predicts(image=image,
                        model=model,
                        delta=delta,
                        patch=patch,
                        combine=combine_horizontal)


def get_vertical_predicts(image, model, delta, patch):
    return get_predicts(image=image,
                        model=model,
                        delta=delta,
                        patch=patch,
                        combine=combine_vertical)
