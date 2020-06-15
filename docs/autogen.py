from keras_autodoc import DocumentationGenerator

pages = {
    'DocClean_API.md': ["docclean.autoencoder",
                        "docclean.autoencoder.build_model",
                        "docclean.autoencoder.train_model",
                        "docclean.CycleGan",
                        "docclean.CycleGan.train",
                        "docclean.CycleGan.train_step",
                        "docclean.utils.get_png_data",
                        "docclean.utils.get_kaggle_paired_data",
                        "docclean.utils.get_kaggle_data",
                        "docclean.utils.normalize",
                        "docclean.utils.books_crop_and_augment",
                        "docclean.utils.kaggle_paired_augment"]
}

doc_generator = DocumentationGenerator(pages)
doc_generator.generate('./sources')
