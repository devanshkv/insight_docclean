import os
import shutil

from keras_autodoc import DocumentationGenerator

pages = {
    'AutoEncoder.md': [
        "docclean.autoencoder.Autoencoder",
        "docclean.autoencoder.Autoencoder.build_model",
        "docclean.autoencoder.Autoencoder.train_model"
    ],
    'CycleGAN.md': [
        "docclean.cycle_gan.CycleGan",
        "docclean.cycle_gan.CycleGan.train"
    ],
    'Utils.md': [
        "docclean.utils.ImageMosaic",
        "docclean.utils.ImageMosaic.get_powers_of_two",
        "docclean.utils.ImageMosaic.normalise",
        "docclean.utils.ImageMosaic.extend_image",
        "docclean.utils.ImageMosaic.make_patches",
        "docclean.utils.ImageMosaic.combine_patches",
        "docclean.utils.normed_to_uint8",
        "docclean.utils.get_png_data",
        "docclean.utils.get_kaggle_paired_data",
        "docclean.utils.get_kaggle_data",
        "docclean.utils.normalize",
        "docclean.utils.books_crop_and_augment",
        "docclean.utils.kaggle_paired_augment"
    ]
}

doc_generator = DocumentationGenerator(pages)
doc_generator.generate('./sources')

shutil.copyfile('../README.md', 'sources/index.md')
os.makedirs(os.path.dirname('sources/data/'), exist_ok=True)
shutil.copyfile('../data/demo.gif', 'sources/data/demo.gif')
