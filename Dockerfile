# Dockerfile for building streamlit app

FROM continuumio/miniconda3
RUN conda install python=3.8

# copy local files into container
COPY . /tmp/

EXPOSE 8888

# change directory
WORKDIR /tmp

# install dependencies
RUN pip install -r requirements.txt
RUN python3 setup.py install

# run commands
CMD ["streamlit", "run", "app.py"]