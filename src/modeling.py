import joblib
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["cleaned_text"]).copy()

    print("Information of ML dataset:")
    print(df.info())
    print(df["sentiment"].value_counts())
    print(df.head())

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 123
):
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    for train_idx, test_idx in sss.split(df, df["sentiment"]):
        train_set = df.iloc[train_idx].copy()
        test_set = df.iloc[test_idx].copy()

    X_train = train_set.drop("sentiment", axis=1)
    X_test = test_set.drop("sentiment", axis=1)
    y_train = train_set["sentiment"].copy()
    y_test = test_set["sentiment"].copy()

    print("Training data sentiment distribution:")
    print(y_train.value_counts())
    print("Testing data sentiment distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )


def vectorize_text(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    vectorizer_path: str = "models/tfidf_vectorizer.pkl"
):
    vectorizer = build_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train["cleaned_text"])
    X_test_tfidf = vectorizer.transform(X_test["cleaned_text"])

    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")
    print("TF-IDF shapes:", X_train_tfidf.shape, X_test_tfidf.shape)

    return vectorizer, X_train_tfidf, X_test_tfidf


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            C=10,
            random_state=42
        ),
        "Linear SVM": LinearSVC(
            C=1,
            max_iter=10000,
            random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            solver="adam",
            random_state=42
        )
    }


def train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test):
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_scaled = scaler.transform(X_test_tfidf)

    models = get_models()

    results = {}
    detailed_metrics = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == "MLP Classifier":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report_text = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        print(f"Accuracy for {name}: {acc:.4f}")
        print(f"Classification Report for {name}:\n{report_text}")

        results[name] = acc
        detailed_metrics[name] = {
            "confusion_matrix": cm,
            "report": report_dict
        }

    print("\nSummary of Model Performances:")
    for model_name, acc in results.items():
        print(f"{model_name}: {acc:.4f}")

    return results, detailed_metrics


def tune_logistic_regression(X_train_tfidf, y_train):
    param_grid_lr = {"C": [0.01, 0.1, 1, 10]}
    lr = LogisticRegression(max_iter=5000, solver="liblinear", random_state=42)

    grid_lr = GridSearchCV(
        lr,
        param_grid_lr,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_lr.fit(X_train_tfidf, y_train)

    print("Best Logistic Regression Parameters:", grid_lr.best_params_)
    print("Best Logistic Regression CV Score:", grid_lr.best_score_)

    return grid_lr


def tune_linear_svm(X_train_tfidf, y_train):
    param_grid_svm = {"C": [0.01, 0.1, 1, 10]}
    svm = LinearSVC(max_iter=10000, random_state=42)

    grid_svm = GridSearchCV(
        svm,
        param_grid_svm,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_svm.fit(X_train_tfidf, y_train)

    print("Best SVM Parameters:", grid_svm.best_params_)
    print("Best SVM CV Score:", grid_svm.best_score_)

    return grid_svm


def evaluate_tuned_model(model, X_test_tfidf, y_test, model_name: str):
    y_pred = model.predict(X_test_tfidf)

    print(f"{model_name} Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def save_tuned_models(
    grid_lr,
    grid_svm,
    logistic_path: str = "models/Logistic_model.pkl",
    svm_path: str = "models/LinearSVM_model.pkl"
):
    joblib.dump(grid_lr, logistic_path)
    joblib.dump(grid_svm, svm_path)

    print(f"Saved Logistic Regression model to: {logistic_path}")
    print(f"Saved Linear SVM model to: {svm_path}")


def run_modeling_pipeline(
    input_csv_path: str = "data/remaining_reviews.csv",
    vectorizer_path: str = "models/tfidf_vectorizer.pkl",
    logistic_model_path: str = "models/Logistic_model.pkl",
    svm_model_path: str = "models/LinearSVM_model.pkl"
):
    df = load_training_data(input_csv_path)
    X_train, X_test, y_train, y_test = split_data(df)

    _, X_train_tfidf, X_test_tfidf = vectorize_text(
        X_train,
        X_test,
        vectorizer_path=vectorizer_path
    )

    results, detailed_metrics = train_and_evaluate_models(
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test
    )

    grid_lr = tune_logistic_regression(X_train_tfidf, y_train)
    grid_svm = tune_linear_svm(X_train_tfidf, y_train)

    evaluate_tuned_model(grid_lr, X_test_tfidf, y_test, "Logistic Regression")
    evaluate_tuned_model(grid_svm, X_test_tfidf, y_test, "Linear SVM")

    save_tuned_models(
        grid_lr,
        grid_svm,
        logistic_path=logistic_model_path,
        svm_path=svm_model_path
    )

    return results, detailed_metrics


if __name__ == "__main__":
    run_modeling_pipeline()
