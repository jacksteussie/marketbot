# Setup the base image
FROM continuumio/miniconda3 as base
SHELL ["/bin/bash", "-c"]
RUN conda update -n base -c defaults conda  
RUN apt-get update && apt-get -y upgrade \
&& apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    gcc \
    ca-certificates \
    make \
    && rm -rf /var/lib/apt/lists/*


# Build ta-lib from source
RUN cd root \
&& wget --show-progress "https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz" \
&& tar -xvzf ta-lib-0.4.0-src.tar.gz ta-lib/ \
&& rm ta-lib-0.4.0-src.tar.gz
RUN cd root/ta-lib \
&& ./configure --build=aarch64-unknown-linux-gnu \
&& make \
&& make install


# Install python dependencies through conda + pip
COPY dependencies.yaml ./
RUN conda env create -n marketbot -f dependencies.yaml
RUN conda init bash && exit
RUN echo "conda activate marketbot" >> ~/.bashrc


# Add the source code
ADD . /marketbot


# Finish up the image
WORKDIR /marketbot
ENTRYPOINT ["python"]
CMD ["main.py"]