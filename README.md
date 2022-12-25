# MarketBot

![Docker Automated build](https://img.shields.io/docker/automated/jacksteussie/marketbot?label=docker%20build&logo=docker)

## Project Vision

MarketBot is a work-in-progress python-based stock/crypto trading bot manager. Various features and support that it will contain include:

- Machine learning and deep learning model training/testing pipelines (specifically supports `sklearn.pipeline` but other pipeline frameworks could probably be pretty easily built in as well).
- Feature download and selection pipelines for the previously mentioned models (i.e. features such as economic data, stock OHLC data, technical analysis (TA) indicators, etc.).
- Trading bot deployment and management framework.
- Bot strategy creation utilizing traditional TA-indicator based models as well as using machine learning models as a predicition method.
- And more!

**Note:** *For all intents and purposes this repository is a work in progress and will be updated over time. Also large overhauls may be done at any given time in the case that the vision of the project changes in its early stages. Thank you for understanding. Also note that this project is in the very early stages of its life, so any feedback/advice is appreciated! Also any collaboration is welcome as well!*

## API SUPPORT

* [ ] Coinbase
* [x] TD Ameritrade
* [x] polygon.io
* [x] FRED

## API CREDS

In order for the package to work and connect to the APIs on
your system, you must put add the file 'creds.ini' to the directory
'/private/'. The final path of the file will look like '/private/creds.ini'. This file should be formatted as follows:
```
[TDA_AUTH]
API_KEY = <Insert TD Ameritrade API key here>
REDIRECT = <Insert redirect url here (for the corresponding TDA API key's application)
TOKEN_PATH = <insert token file path here (probably root of project)>
ACCOUNT_ID = <Insert TD Ameritrade account id here>

[CB_AUTH]
API_KEY = <Insert Coinbase API key here>
API_SECRET = <Insert Coinbase API secret key here>

[POLY_AUTH]
API_KEY = <Insert polygon.io API key here>
```

## HOW TO INSTALL WITH DOCKER

In order to install and run this project to a container follow these steps:

1. Using a terminal, clone this repository to a directory on your system of choice: \
docker build github.com/jacksteussie/marketbot

2. Do the same thing with the API credentials and ```creds.ini``` as described above but in the docker container running the built image.

3. Keep in mind, if developing/running tests in the container, you will have to initialize the conda environment inside bash. So when you get into the container and execute a bash shell, type ```conda activate marketbot```.