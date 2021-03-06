import argparse
import glob
import logging

import pylab as plt
import tensorflow as tf
import tqdm

import docclean

logger = logging.getLogger(__name__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Training Script for DocClean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--type",
        help="Which model to train",
        choices=["cycle_gan", "autoencoder"],
        required=True,
        type=str,
    )
    parser.add_argument(
        "-k", "--kaggle_data_dir", help="Kaggle Data Directory", required=True
    )
    parser.add_argument(
        "-c",
        "--clean_books_dir",
        help="Directory containing clean images",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dirty_books_dir",
        help="Directory containing dirty images",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs to train for",
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size", type=int, default=16, required=False
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")

    args = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    # Set up kaggle data ingestion

    kaggle_img_list = glob.glob(f"{args.kaggle_data_dir}/*png")
    train_imgs_list = []
    logging.info("Reading Kaggle Data.")
    for img in tqdm.tqdm(kaggle_img_list):
        im_frame = plt.imread(img)
        if im_frame.shape[0] != 258:
            train_imgs_list.append(img)

    clean_imgs_list = [x.replace("train", "train_cleaned") for x in train_imgs_list]

    if args.type == "autoencoder":
        list_ds = tf.data.Dataset.from_tensor_slices(train_imgs_list)
        list_ds = list_ds.shuffle(buffer_size=96, seed=1993)
        train_dataset = list_ds.take(80)
        validate_dataset = list_ds.skip(80)

        list_ds = train_dataset.repeat(15)
        validate_dataset = validate_dataset.repeat(10)

        labeled_ds = list_ds.map(
            docclean.utils.get_kaggle_paired_data, num_parallel_calls=AUTOTUNE
        ).cache()
        val_ds = validate_dataset.map(
            docclean.utils.get_kaggle_paired_data, num_parallel_calls=AUTOTUNE
        ).cache()

        labeled_ds = labeled_ds.map(
            docclean.utils.kaggle_paired_augment, num_parallel_calls=AUTOTUNE
        )

        val_ds = val_ds.map(
            docclean.utils.kaggle_paired_augment, num_parallel_calls=AUTOTUNE
        )

        labeled_ds = labeled_ds.batch(args.batch_size)
        labeled_ds = labeled_ds.prefetch(AUTOTUNE)

        val_ds = val_ds.batch(args.batch_size)
        val_ds = val_ds.prefetch(AUTOTUNE)

        logging.info("Creating Autoencoder Model")
        autoencoder = docclean.autoencoder.Autoencoder()
        logging.info(f"Training Autoencoder Model for {args.epochs} epochs.")
        autoencoder.train_model(labeled_ds, validation_data=val_ds, epochs=args.epochs)
        logging.info(f"Saving the model under the name Docclean_autoencoder.")
        autoencoder.autoencoder_model.save_weights("weights/ae")

    else:
        kaggle_dirty_images = tf.data.Dataset.from_tensor_slices(train_imgs_list)
        kaggle_clean_images = tf.data.Dataset.from_tensor_slices(clean_imgs_list)

        kaggle_dirty_images = kaggle_dirty_images.shuffle(buffer_size=96, seed=1993)
        kaggle_clean_images = kaggle_clean_images.shuffle(buffer_size=96, seed=1993)

        kaggle_dirty_images = kaggle_dirty_images.take(80)
        kaggle_clean_images = kaggle_clean_images.take(80)
        kaggle_dirty_images = kaggle_dirty_images.repeat(10)
        kaggle_clean_images = kaggle_clean_images.repeat(10)

        kaggle_dirty_images = kaggle_dirty_images.map(
            docclean.utils.get_kaggle_data, num_parallel_calls=AUTOTUNE
        ).cache()
        kaggle_clean_images = kaggle_clean_images.map(
            docclean.utils.get_kaggle_data, num_parallel_calls=AUTOTUNE
        ).cache()

        kaggle_dirty_images = kaggle_dirty_images.map(
            docclean.utils.kaggle_crop_and_augment, num_parallel_calls=AUTOTUNE
        )
        kaggle_clean_images = kaggle_clean_images.map(
            docclean.utils.kaggle_crop_and_augment, num_parallel_calls=AUTOTUNE
        )

        if args.dirty_books_dir is not None:
            logging.info("Reading Dirty/Clean books data.")
            books_dirty_images = tf.data.Dataset.from_tensor_slices(
                glob.glob(f"{args.dirty_books_dir}/*png")
            ).shuffle(buffer_size=40960)
            books_clean_images = tf.data.Dataset.from_tensor_slices(
                glob.glob(f"{args.clean_books_dir}/*png")
            ).shuffle(buffer_size=40960)
            books_dirty_images = books_dirty_images.map(
                docclean.utils.get_png_data
            ).cache()
            books_clean_images = books_clean_images.map(
                docclean.utils.get_png_data
            ).cache()
            books_dirty_images = books_dirty_images.map(
                docclean.utils.books_crop_and_augment, num_parallel_calls=AUTOTUNE
            )
            books_clean_images = books_clean_images.map(
                docclean.utils.books_crop_and_augment, num_parallel_calls=AUTOTUNE
            )

            dirty_images = tf.data.Dataset.concatenate(
                kaggle_dirty_images, books_dirty_images
            )
            clean_images = tf.data.Dataset.concatenate(
                kaggle_clean_images, books_clean_images
            )

        else:
            dirty_images = kaggle_dirty_images
            clean_images = kaggle_clean_images

        dirty_images = (
            dirty_images.shuffle(4096).batch(args.batch_size).prefetch(AUTOTUNE)
        )
        clean_images = (
            clean_images.shuffle(4096).batch(args.batch_size).prefetch(AUTOTUNE)
        )

        logging.info("Creating the Cycle GAN model.")
        cycle_gan = docclean.cycle_gan.CycleGan()

        logging.info("Training the Cycle GAN model.")
        cycle_gan.train(dirty_images, clean_images, epochs=args.epochs)
        logging.info("Saving the model under Docclean_cyclegan")
        cycle_gan.generator_g.save_weights("weights/cg")
