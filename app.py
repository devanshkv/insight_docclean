import numpy as np
import streamlit as st
from PIL import Image

try:
    import pytesseract
    import pytesseract.TesseractNotFoundError as TesseractNotFoundError

    tesseract_available = True
except ImportError:
    tesseract_available = False

import docclean
from docclean.utils import ImageMosaic

st.write("# Docclean")
uploaded_file = st.file_uploader("Upload an image file")

if uploaded_file:

    input_file = Image.open(uploaded_file).convert('RGB')
    input_file = np.asarray(input_file)

    st.write("### Uploaded Image:")
    st.image(input_file)
    st.write("### Choose a Model:")
    model = st.selectbox("", ('Vanilla', 'CycleGAN'))

    if model == 'CycleGAN':
        from tensorflow_examples.models.pix2pix import pix2pix

        model = pix2pix.unet_generator(3, norm_type='instancenorm')
        model.load_weights("weights/cg")
    else:
        model = docclean.autoencoder.Autoencoder().autoencoder_model
        model.load_weights("weights/ae")

    im = ImageMosaic(input_file)
    batches = im.make_patches()

    if model == 'CycleGAN':
        batches = 2 * batches - 1

    out = model.predict(batches)

    if model == 'CycleGAN':
        out = (out + 1) / 2

    output_image = im.combine_patches(out)

    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    st.image(output_image)
    output_image *= 255

    if tesseract_available:

        try:
            string = pytesseract.image_to_string(Image.fromarray(output_image.astype('uint8'), 'RGB'))
            st.write("### Output from Tessaract:")
            st.write(string)
        except TesseractNotFoundError:
            pass
