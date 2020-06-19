# DocClean
[![GitHub issues](https://img.shields.io/github/issues/devanshkv/insight_docclean?style=flat-square)](https://github.com/devanshkv/insight_docclean/issues)
[![GitHub forks](https://img.shields.io/github/forks/devanshkv/insight_docclean?style=flat-square)](https://github.com/devanshkv/insight_docclean/network/members)
[![GitHub stars](https://img.shields.io/github/stars/devanshkv/insight_docclean?style=flat-square)](https://github.com/devanshkv/insight_docclean/stargazers)
[![GitHub license](https://img.shields.io/github/license/devanshkv/insight_docclean?style=flat-square)](https://github.com/devanshkv/insight_docclean/blob/master/LICENSE)
[![HitCount](http://hits.dwyl.com/devanshkv/insight_docclean.svg)](http://hits.dwyl.com/devanshkv/insight_docclean)
[![codecov](https://codecov.io/gh/devanshkv/insight_docclean/branch/master/graph/badge.svg?style=flat-square)](https://codecov.io/gh/devanshkv/insight_docclean)
![Python application](https://github.com/devanshkv/insight_docclean/workflows/Python%20application/badge.svg?style=flat-square)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2Fdevanshkv%2Finsight_docclean)](https://twitter.com/devanshkv)


Clean your document images with crumpled backgrounds, strains and folds with Deep Neural Networks.


Here is a demo:

![](data/demo.gif)

You can find the slide deck accompanying this project [here](https://docs.google.com/presentation/d/1k0ulZ-9ExgH9h1v2684A7HlRs-bKhIFVl0FxAtLuRds/edit?usp=sharing). 

## Installation
For installing docclean is easy. Just run the following:

```
git clone https://github.com/devanshkv/insight_docclean.git
cd insight_docclean
pip install -r requirements.txt
python3 setup.py install
```

## Documentation
Have a look at our beauitful docs [here](https://devanshkv.github.io/insight_docclean/).

## Training the models
The models can be trained using `train.py`. The usage is as follows:

```
usage: train.py [-h] -t {cycle_gan,autoencoder} -k KAGGLE_DATA_DIR
                     [-c CLEAN_BOOKS_DIR]
                     [-d DIRTY_BOOKS_DIR] [-e EPOCHS]
                     [-b BATCH_SIZE] [-v]
```
### Quick reference table
|Short|Long               |Default|Description                      |
|-----|-------------------|-------|---------------------------------|
|`-h` |`--help`           |       |show this help message and exit  |
|`-t` |`--type`           |`None` |Which model to train             |
|`-k` |`--kaggle_data_dir`|`None` |Kaggle Data Directory            |
|`-c` |`--clean_books_dir`|`None` |Directory containing clean images|
|`-d` |`--dirty_books_dir`|`None` |Directory containing dirty images|
|`-e` |`--epochs`         |`100`  |Number of epochs to train for    |
|`-b` |`--batch_size`     |`16`   |Batch size                       |
|`-v` |`--verbose`        |       |Be verbose                       |

## Running the inference

Using the trained model the infence can be run using `infer.py`. The usage is as follows:

```
usage: infer.py [-h] [-v] [-g GPU_ID] -c DATA_DIR [-b BATCH_SIZE] -t
               {cycle_gan,autoencoder} -w WEIGHTS
```
### Quick reference table
|Short|Long          |Default|Description                    |
|-----|--------------|-------|-------------------------------|
|`-h` |`--help`      |       |show this help message and exit|
|`-v` |`--verbose`   |       |Be verbose                     |
|`-g` |`--gpu_id`    |`0`    |GPU ID (use -1 for CPU)        |
|`-c` |`--data_dir`  |`None` |Directory with candidate pngs. |
|`-b` |`--batch_size`|`32`   |Batch size for training data   |
|`-t` |`--type`      |`None` |Which model to train           |
|`-w` |`--weights`   |`None` |Model weights                  |

## Running the [streamlit](https://www.streamlit.io/) app
Run,
```
streamlit run app.py
```
 and the use `localhost:8501` to view the app.
