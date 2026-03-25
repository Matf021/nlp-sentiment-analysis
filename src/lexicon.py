import joblib
import nltk
import pandas as pd

from textblob import TextBlob, Word
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def download_nltk_resources():
    nltk.download("wordnet")
    nltk.download("vader_lexicon")


def load_lexicon_dataset(path: str = "data/lexicon_reviews.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Check the distribution for testing set")
    print(df["sentiment"].value_counts())
    return df


def load_saved_ml_artifacts(
    vectorizer_path: str = "models/tfidf_vectorizer.pkl",
    logistic_model_path: str = "models/Logistic_model.pkl",
    svm_model_path: str = "models/LinearSVM_model.pkl"
):
    vectorizer = joblib.load(vectorizer_path)
    grid_lr = joblib.load(logistic_model_path)
    grid_svm = joblib.load(svm_model_path)
    return vectorizer, grid_lr, grid_svm


def prepare_ml_test_features(test_set: pd.DataFrame, vectorizer):
    y_test = test_set["sentiment"].copy()
    X_test = test_set.drop("sentiment", axis=1)
    X_test_tfidf = vectorizer.transform(X_test["cleaned_text"])
    return X_test, y_test, X_test_tfidf


def evaluate_ml_models_on_lexicon_set(test_set: pd.DataFrame, vectorizer, grid_lr, grid_svm):
    _, y_test, X_test_tfidf = prepare_ml_test_features(test_set, vectorizer)

    models = {
        "Logistic Regression": grid_lr,
        "Linear SVM": grid_svm
    }

    results = {}
    detailed_metrics = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        print(f"Accuracy for {name}: {acc:.4f}")
        print(f"Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred))

        results[name] = acc
        detailed_metrics[name] = {
            "confusion_matrix": cm,
            "report": report_dict
        }

    ml_metrics = {}
    for model_name, metrics_dict in detailed_metrics.items():
        report = metrics_dict["report"]
        ml_metrics[model_name] = {
            "Accuracy": report["accuracy"],
            "Precision": report["macro avg"]["precision"],
            "Recall": report["macro avg"]["recall"],
            "F1-score": report["macro avg"]["f1-score"]
        }

    return results, detailed_metrics, ml_metrics


def lemmatize_text(text: str) -> str:
    blob = TextBlob(text)
    lemmatized = " ".join([Word(word).lemmatize() for word in blob.words])
    return lemmatized


def correct_text(text: str) -> str:
    corrected = TextBlob(text).correct()
    return str(corrected)


def classify_textblob_sentiment(polarity: float) -> str:
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


def run_textblob(test_set: pd.DataFrame):
    test_set = test_set.copy()

    test_set["lemmatized_text"] = test_set["cleaned_text"].apply(lemmatize_text)
    test_set["corrected_text"] = test_set["lemmatized_text"].apply(correct_text)
    test_set["tb_polarity"] = test_set["corrected_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    test_set["sentiment_textblob"] = test_set["tb_polarity"].apply(classify_textblob_sentiment)

    accuracy = accuracy_score(test_set["sentiment"], test_set["sentiment_textblob"])
    print(f"TextBlob Accuracy: {accuracy * 100:.2f}%\n")
    print("TextBlob Classification Report:")
    print(classification_report(test_set["sentiment"], test_set["sentiment_textblob"]))

    report = classification_report(
        test_set["sentiment"],
        test_set["sentiment_textblob"],
        output_dict=True
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision": report["macro avg"]["precision"],
        "Recall": report["macro avg"]["recall"],
        "F1-score": report["macro avg"]["f1-score"]
    }

    return test_set, metrics


def classify_vader_sentiment(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def run_vader(test_set: pd.DataFrame):
    test_set = test_set.copy()

    sia = SentimentIntensityAnalyzer()

    test_set["sentiment_scores"] = test_set["cleaned_text"].apply(lambda x: sia.polarity_scores(x))
    test_set["compound"] = test_set["sentiment_scores"].apply(lambda x: x["compound"])
    test_set["positive"] = test_set["sentiment_scores"].apply(lambda x: x["pos"])
    test_set["neutral"] = test_set["sentiment_scores"].apply(lambda x: x["neu"])
    test_set["negative"] = test_set["sentiment_scores"].apply(lambda x: x["neg"])
    test_set["sentiment_vader"] = test_set["compound"].apply(classify_vader_sentiment)

    print(test_set[["cleaned_text", "compound", "sentiment", "sentiment_vader"]].head())

    accuracy = accuracy_score(test_set["sentiment"], test_set["sentiment_vader"])
    print(f"VADER Accuracy: {accuracy * 100:.2f}%")
    print("VADER Classification Report:")
    print(classification_report(test_set["sentiment"], test_set["sentiment_vader"]))

    report = classification_report(
        test_set["sentiment"],
        test_set["sentiment_vader"],
        output_dict=True
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision": report["macro avg"]["precision"],
        "Recall": report["macro avg"]["recall"],
        "F1-score": report["macro avg"]["f1-score"]
    }

    return test_set, metrics


def build_comparison_dataframe(ml_metrics: dict, textblob_metrics: dict, vader_metrics: dict) -> pd.DataFrame:
    all_metrics = {
        "TextBlob": textblob_metrics,
        "VADER": vader_metrics,
        **ml_metrics
    }

    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df[["Accuracy", "Precision", "Recall", "F1-score"]]

    print("Comparison of Performance Metrics:")
    print(comparison_df)

    return comparison_df


def run_lexicon_pipeline(
    test_path: str = "data/lexicon_reviews.csv",
    vectorizer_path: str = "models/tfidf_vectorizer.pkl",
    logistic_model_path: str = "models/Logistic_model.pkl",
    svm_model_path: str = "models/LinearSVM_model.pkl"
):
    download_nltk_resources()

    test_set = load_lexicon_dataset(test_path)
    vectorizer, grid_lr, grid_svm = load_saved_ml_artifacts(
        vectorizer_path=vectorizer_path,
        logistic_model_path=logistic_model_path,
        svm_model_path=svm_model_path
    )

    _, _, ml_metrics = evaluate_ml_models_on_lexicon_set(test_set, vectorizer, grid_lr, grid_svm)

    textblob_test_set, textblob_metrics = run_textblob(test_set)
    vader_test_set, vader_metrics = run_vader(test_set)

    comparison_df = build_comparison_dataframe(
        ml_metrics=ml_metrics,
        textblob_metrics=textblob_metrics,
        vader_metrics=vader_metrics
    )

    return {
        "comparison_df": comparison_df,
        "textblob_results": textblob_test_set,
        "vader_results": vader_test_set,
        "ml_metrics": ml_metrics,
        "textblob_metrics": textblob_metrics,
        "vader_metrics": vader_metrics
    }


if __name__ == "__main__":
    run_lexicon_pipeline()
