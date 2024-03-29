FROM continuumio/miniconda3:4.8.2

LABEL Name=pareidolia Version=1.2.0

COPY * ./ /app/
WORKDIR /app

RUN apt-get update && apt-get install -y gcc
RUN conda config --add channels bioconda
RUN conda install -c conda-forge -y cooler pip \
  && conda clean -afy

RUN pip install -Ur requirements.txt
RUN pip install .

ENTRYPOINT [ "pareidolia" ]
