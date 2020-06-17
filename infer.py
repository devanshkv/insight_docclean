import argparse
import glob
import logging
import os

import tensorflow as tf
import tqdm
import numpy as np
from PIL import Image

import docclean

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocClean", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-g', '--gpu_id', help='GPU ID (use -1 for CPU)', type=int, required=False, default=0)
    parser.add_argument('-c', '--data_dir', help='Directory with candidate pngs.', required=True, type=str)
    parser.add_argument('-b', '--batch_size', help='Batch size for training data', default=32, type=int)
    parser.add_argument('-t', '--type', help='Which model to train', choices=['cycle_gan', 'autoencoder'],
                        required=True, type=str)
    parser.add_argument('-w', '--weights', help='Model weights', required=True, type=str)
    args = parser.parse_args()

    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu_id}'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cands_to_eval = glob.glob(f'{args.data_dir}/*png')

    if len(cands_to_eval) == 0:
        raise FileNotFoundError(f"No candidates to evaluate.")

    if args.type == 'cycle_gan':
        from tensorflow_examples.models.pix2pix import pix2pix

        model = pix2pix.unet_generator(3, norm_type='instancenorm')

    else:
        model = docclean.autoencoder.Autoencoder().autoencoder_model

    model.load_weights(args.weights)

    input_img_list = glob.glob(f"{args.data_dir} + '/*png")

    list_ds = tf.data.Dataset.from_tensor_slices(input_img_list)
    infer_images = list_ds.map(docclean.utils.get_png_data, num_parallel_calls=AUTOTUNE)
    infer_images = infer_images.map(docclean.normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    inffered_images = model.predict(infer_images, verbose=1, batch_size=args.batch_size)

    if not isinstance(inffered_images, np.ndarray):
        inffered_images = inffered_images.numpy()

    for idx, img_name in tqdm.tqdm(enumerate(input_img_list)):
        outname = img_name[:-4] + '_cleaned.png'
        im = Image.fromarray(inffered_images[idx], 'RGB')
        im.save(outname)
