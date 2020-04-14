# Disaster_Response_Pipeline
This repository contains the code for a disaster response pipeline, a project for my Udacity data science course.
Objective is to write an interactive web-app that allows the classification of messages into different categories
e.g. request for aid, medical help etc. A pre labeld data set from FigureEight is utilized as training data, for 
the following data processing steps:

- Data preproessing (Extract Transform Load)
    - Extraction of data from csv, transformation usable tabular form, loading into sql database
- Training and optimization of data machine learning pipeline
    - Data is loaded form sql database and divided into features and labels
    - Features (written text) is cleaned, tokenized, lemmatized and transformed into Tfidf representation
    - Length of the text message is used as an additional feature
    - Data is classified using a multi output Random Forrest Classifier
    - The whole data pipeline is optimized using a grid search, optimizing with respect to the number of decision trees for the classifier
    - The optimized pipeline is saves as a pickle


# File and folder description
File/Folder| Description 
--- | ---
...| .....
.... | ...

# Installation 
```bash
pip install -r requirements.txt
```

## If you can want to run the file in a new enviroment:
- Make sure conda is installed (Best practice, set up with virtualenv is not tested)
- Open a terminal or a anaconda prompt
- If desired make new enviroment: conda create -n name_of_enviroment python
- Activate enviroment conda activate: conda create name_of_enviroment
- Install dependencies: pip install requirements.txt
- If the new enviroment / kernel is supposed to be used in Jupyter, install kernel:
```bash
    python -m ipykernel install --name name_of_enviroment
```
- Open your Jupyter Notebook it should work now


