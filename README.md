# Disaster Response Pipeline Project

## Table of Contents

* [Overview](#project-overview)
* [Requirements](#requirements)
* [Components](#components)
* [Instructions](#instructions)
* [Acknowledgements](#acknowledgements)

## Overview

In this project, I applied my data engineering skills to processe data and build a model trained for an API that classify disaster text data. The project include a machine learning pipeline and a web interface for demo where user can input a messenge and get classification result.

## Requirements

Requirements libraries for this project is listed below, which can be install via pip.

* pandas
* sqlalchemy
* nltk
* sklearn
* pickle
* flask
* json
* plotly
* joblib



## Components

This project has three components:

1. `ETL Pipeline`
    * A Python script "process_data.py" to load data from csv file, preprocess data and store it in SQLite database.
    * Two data files: disaster_categories.csv, disaster_messages.csv

2. `ML Pipeline`
    * A Python script "train_classifier.py" writes a machine learning pipeline to load data from SQLite database, spit data into train and test dataset, train model and export model as pickel file.

3. `Flask Web App`
    * A web app to demo how messenge can be classification.
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app after access app directory (`cd app`).
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
Thanks Udacity for helping me complete this project.