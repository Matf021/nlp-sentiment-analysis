# NLP Sentiment Analysis: Lexicon vs Machine Learning

This project analyzes Amazon product reviews using both lexicon based and machine learning approaches for sentiment classification. It compares traditional NLP methods such as VADER and TextBlob against supervised machine learning models built with TF-IDF features.

In addition to the baseline comparison, the project also includes an advanced extension using Aspect Based Sentiment Analysis (ABSA) to explore fine grained sentiment and rating adjustment.

## Project Overview

The main goal of this project is to evaluate how well different sentiment analysis approaches perform on product reviews.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Text preprocessing and sentiment labeling
- TF-IDF feature extraction
- Supervised machine learning models
- Lexicon-based sentiment analysis with VADER and TextBlob
- Comparative evaluation using accuracy, precision, recall, and F1-score
- Aspect Based Sentiment Analysis (ABSA) as an advanced extension

## Methods Used

### 1. Data Exploration
The dataset was explored to understand:
- rating distribution
- helpful vote behavior
- review length patterns
- product and user review distributions
- duplicate and outlier reviews

### 2. Preprocessing
The preprocessing pipeline includes:
- duplicate removal
- filtering non-English reviews
- combining title and review text
- cleaning and normalizing text
- generating sentiment labels from ratings
- removing review-length outliers

Sentiment labels were assigned as:
- **Positive** for ratings 4–5
- **Neutral** for rating 3
- **Negative** for ratings 1–2

### 3. Machine Learning Models
TF-IDF was used for text representation, with unigrams and bigrams.

The following models were trained and evaluated:
- Logistic Regression
- Linear SVM
- Naive Bayes
- Gradient Boosting
- MLP Classifier

Hyperparameter tuning was also performed for:
- Logistic Regression
- Linear SVM

### 4. Lexicon-Based Methods
Two lexicon-based approaches were tested:
- **TextBlob**
- **VADER**

These were compared directly against the trained machine learning models.

### 5. ABSA Extension
An Aspect Based Sentiment Analysis module was included as an advanced extension to:
- extract product aspects from reviews
- analyze sentiment around individual aspects
- detect mixed sentiment and rating-text misalignment
- explore adjusted ratings based on aspect-level sentiment
