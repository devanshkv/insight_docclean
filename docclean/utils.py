from typing import Tuple

import numpy as np
import tensorflow as tf


def get_png_data(fname: str) -> tf.Tensor:
    """
    Read png data into tf tensors.

    Args:

        fname (str): file path

    Returns:

        tf.Tensor: image tensor

    """
    img = tf.io.read_file(fname)
    img = tf.io.decode_png(img, dtype=tf.uint8)
    return img


def get_kaggle_paired_data(fname: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Get kaggle paired data

    Args:

        fname (str): File Name

    Returns:

       Tuple [tf.Tensor, tf.Tensor] : Dirty and Claen image

    """
    img = tf.io.read_file(fname)
    train_img = tf.io.decode_png(img, dtype=tf.uint8)
    fname = tf.strings.regex_replace(fname, "train", "train_cleaned")
    img = tf.io.read_file(fname)
    predict_img = tf.io.decode_png(img, dtype=tf.uint8)
    img, predict_img = tf.cast(train_img, tf.float32) / 255, tf.cast(predict_img, tf.float32) / 255
    img = tf.image.resize_with_pad(img, 512, 512, antialias=True)
    predict_img = tf.image.resize_with_pad(predict_img, 512, 512, antialias=True)
    return tf.concat([img] * 3, axis=-1), tf.concat([predict_img] * 3, axis=-1)


def get_kaggle_data(fname):
    """
    Read kaggle png data into tf tensors.

    Args:

        fname (str): file path

    Returns:

        tf.Tensor: image tensor

    """
    img = tf.io.read_file(fname)
    img = tf.io.decode_png(img, dtype=tf.uint8)
    return tf.concat([img] * 3, axis=-1)


def normalize(image: tf.Tensor) -> tf.Tensor:
    """
    Normalise the image by casting it to float and scaling between -1 and 1

    Args:

        image (tf.Tensor): imgae tensor

    Returns:

        tf.Tensor: Normalised image

    """
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def books_crop_and_augment(image: tf.Tensor, size: Tuple[int, int] = (256, 256), num_boxes: int = 1,
                           rotate: bool = True, flips: bool = True) -> tf.Tensor:
    """
    Augments the book pages by zooming, cropping, rotating and fliiping

    Args:

        image (tf.Tensor) : image tensor

        size (Tuple) : size to crop the image tensor

        num_boxes (int): Number of patches from the image

        rotate (bool): If random 90 degree rotations

        flips (bool): If random LR and UD flips

    Returns:

        tf.Tensor: Augmented image

    """
    combined = tf.image.central_crop(image, central_fraction=0.7)
    combined = normalize(combined)
    combined = tf.expand_dims(combined, axis=0)

    scale = tf.random.uniform(minval=0.2, maxval=0.5, shape=(num_boxes,))
    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes = tf.convert_to_tensor([x1, y1, x2, y2])
    boxes = tf.reshape(boxes, shape=(num_boxes, 4))
    box_indices = tf.random.uniform(shape=(num_boxes,), minval=0, maxval=num_boxes, dtype=tf.int32)
    combined = tf.image.crop_and_resize(combined, boxes, box_indices, size)

    if rotate:
        combined = tf.image.rot90(combined, k=np.random.randint(0, 4))
    if flips:
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)

    return combined[0]


def kaggle_crop_and_augment(image: tf.Tensor, size: Tuple[int, int] = (256, 256), rotate: bool = True,
                            flips: bool = True) -> tf.Tensor:
    """
    Augments the book pages by zooming, cropping, rotating and fliiping

    Args:

        image (tf.Tensor) : image tensor

        size (Tuple) : size to crop the image tensor

        rotate (bool): If random 90 degree rotations

        flips (bool): If random LR and UD flips

    Returns:

        tf.Tensor: Augmented image

    """
    combined = normalize(image)
    combined = tf.image.random_crop(combined, size=[*size, 3])

    if rotate:
        combined = tf.image.rot90(combined, k=np.random.randint(0, 4))
    if flips:
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)

    return combined


def kaggle_paired_augment(dirty: tf.Tensor, clean: tf.Tensor, size: Tuple[int, int] = (256, 256), rotate: bool = True,
                          flips: bool = True) -> tf.Tensor:
    """
    Augments the book pages by zooming, cropping, rotating and fliiping

    Args:

        dirty (tf.Tensor) : image tensor

        clean (tf.Tensor) : image tensor

        size (Tuple) : size to crop the image tensor

        rotate (bool): If random 90 degree rotations

        flips (bool): If random LR and UD flips

    Returns:

        tf.Tensor: Augmented image

    """
    combined = tf.concat([dirty, clean], axis=-1)
    combined = tf.image.random_crop(combined, size=[*size, 6])

    if rotate:
        combined = tf.image.rot90(combined, k=np.random.randint(0, 4))

    if flips:
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)

    return combined[:, :, :3], combined[:, :, 3:]
