import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
import langid

DetectorFactory.seed = 0


def safe_detect(text: str) -> str:
    try:
        text = text.strip()
        if len(text) < 3:
            return "unknown"
        return detect(text)
    except (LangDetectException, AttributeError):
        return "unknown"


def normalize_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(
        lambda x: x.encode("ascii", "ignore").decode("utf-8") if isinstance(x, str) else x
    )
    df["text"] = df["text"].fillna("No review")
    df["title"] = df["title"].fillna("")
    return df


def detect_languages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lang_detect"] = df["text"].apply(safe_detect)
    df["lang_langid"] = df["text"].apply(
        lambda x: langid.classify(x)[0] if isinstance(x, str) and x.strip() else "unknown"
    )
    return df


def remove_duplicates_and_spanish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_count = len(df)

    df = df.drop_duplicates(subset=["user_id", "asin", "text"], keep="first")

    df = df[
        ~(
            (df["lang_langid"] == "es") &
            (df["lang_detect"] == "es")
        )
    ]

    print(f"Removed {initial_count - len(df)} duplicate/non-English reviews")
    return df


def assign_sentiment_from_rating(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df["rating"].apply(
        lambda x: "Positive" if x >= 4 else "Neutral" if x == 3 else "Negative"
    )
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["full_text"] = df[["title", "text"]].fillna("").agg(" ".join, axis=1)

    df["cleaned_text"] = (
        df["full_text"]
        .str.lower()
        .str.replace(r"http\S+", "", regex=True)
        .str.replace(r"<br />", "", regex=True)
        .str.replace(r"[^a-zA-Z0-9\s.,!?;:()\'\"-]", "", regex=True)
        .str.replace(r"\d+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df


def remove_length_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_length"] = df["cleaned_text"].str.len()

    q1, q3 = df["review_length"].quantile([0.25, 0.75])
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df = df[df["review_length"].between(lower_bound, upper_bound)].copy()
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_text_column(df)
    df = detect_languages(df)
    df = remove_duplicates_and_spanish(df)
    df = assign_sentiment_from_rating(df)
    df = clean_text(df)
    df = remove_length_outliers(df)

    print(f"Final preprocessed size: {len(df)}")
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts())

    return df[["cleaned_text", "sentiment"]].copy()


def save_preprocessed_data(
    input_csv_path: str,
    output_preprocessed_path: str = "data/preprocessed_data.csv",
    output_lexicon_path: str = "data/lexicon_reviews.csv",
    output_remaining_path: str = "data/remaining_reviews.csv",
    lexicon_sample_size: int = 1000,
    random_state: int = 42
) -> None:

    df = pd.read_csv(input_csv_path)
    preprocessed_df = preprocess_data(df)

    preprocessed_df.to_csv(output_preprocessed_path, index=False)
    print(f"Saved preprocessed data to: {output_preprocessed_path}")

    lexicon_df = preprocessed_df.sample(
        n=min(lexicon_sample_size, len(preprocessed_df)),
        random_state=random_state
    )
    lexicon_df.to_csv(output_lexicon_path, index=False)
    print(f"Saved lexicon sample to: {output_lexicon_path}")

    remaining_df = preprocessed_df.drop(lexicon_df.index)
    remaining_df.to_csv(output_remaining_path, index=False)
    print(f"Saved remaining ML data to: {output_remaining_path}")

    print(f"Final for Lexicon size: {len(lexicon_df)}")
    print(f"Final for ML training size: {len(remaining_df)}")


if __name__ == "__main__":
    save_preprocessed_data("data/subset_data.csv")
