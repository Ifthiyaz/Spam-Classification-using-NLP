# Spam-Classification-using-NLP
"Spam Classification using NLP" builds SMS spam classifier using text preprocessing, TF-IDF vectorization, and a Naive Bayes model. It processes SMS data by cleaning, tokenizing, and stemming words, then trains the classifier to detect spam. Stages include data loading, preprocessing, training, evaluation to assess accuracy and model performance.

This project focuses on building a simple spam classifier using Natural Language Processing (NLP) and Machine Learning techniques. The classifier is trained to distinguish between spam and non-spam (ham) SMS messages. 

## Project Overview
Spam detection is an important application in NLP, helping to filter unwanted messages. This project utilizes the Naive Bayes algorithm for classification, a popular choice for text classification problems.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project locally, ensure you have the following dependencies installed:
- Python 3.x
- NLTK
- Pandas

Dataset
The dataset used in this project is a collection of SMS messages labeled as spam or ham (non-spam). The dataset should be saved in your Google Drive for access within Google Colab.

Data Preprocessing
The preprocessing steps include:

Data Loading: Loading the dataset from Google Drive.
Cleaning the Data: Dropping unnecessary columns and renaming relevant columns for readability.
Removing Duplicates: Ensuring each message is unique by dropping duplicates.
Tokenization and Text Cleaning:
Converting text to lowercase.
Tokenizing words and removing non-alphanumeric characters.
Removing stopwords (common words like 'the', 'and' that don’t contribute to classification).
Stemming words (reducing words to their root form) using Porter Stemmer.
Model Training
Text Vectorization: Using TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text messages into numerical format.
Splitting the Data: Dividing data into training and test sets.
Training the Model: Using a Multinomial Naive Bayes classifier, which is effective for text classification.
Evaluation
After training the model, it’s evaluated on the test set using accuracy as a metric to measure its performance.

Results
The classifier outputs an accuracy score, indicating how well the model can distinguish between spam and non-spam messages.

Accuracy Score: Display the model's accuracy score here.

Example Output:

Add images or screenshots of model output here

Usage
To use this project:

Clone the repository.
Place the dataset in the appropriate folder.
Run the notebook on Google Colab or a similar environment with access to Google Drive.
  
- Scikit-Learn

You can install the required packages by running:
```bash
pip install pandas nltk scikit-learn

