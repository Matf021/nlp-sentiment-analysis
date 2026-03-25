import re
import pandas as pd
import nltk
import spacy
from datetime import datetime
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<br />", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:()\'\"-]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text.strip()


def load_spacy_model():
    return spacy.load("en_core_web_sm")


def classify_aspect_sentiment(score: float) -> str:
    if score >= 0.2:
        return "Positive"
    elif score <= -0.2:
        return "Negative"
    else:
        return "Neutral"


def extract_aspects(text: str, nlp, stop_words_set) -> list:
    if not isinstance(text, str) or text.strip() == "":
        return []

    aspects = []
    doc = nlp(text)

    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:
            aspects.append(chunk.text.lower())

    for token in doc:
        if token.pos_ == "NOUN":
            modifiers = [child.text.lower() for child in token.children if child.pos_ == "ADJ"]
            if modifiers:
                for mod in modifiers:
                    aspects.append(f"{mod} {token.text.lower()}")

    sentiment_words = [
        "great", "good", "bad", "terrible", "excellent", "poor",
        "awesome", "horrible", "amazing", "awful", "nice",
        "love", "hate", "best", "worst"
    ]

    for token in doc:
        if token.text.lower() in sentiment_words or token.lemma_.lower() in sentiment_words:
            for child in token.children:
                if child.pos_ == "NOUN":
                    aspects.append(child.text.lower())
            if token.head.pos_ == "NOUN":
                aspects.append(token.head.text.lower())

    filtered_aspects = [
        aspect for aspect in aspects
        if aspect not in stop_words_set and len(aspect) > 1
    ]

    return list(set(filtered_aspects))


def extract_aspect_contexts(text: str, aspects: list, nlp, window_size: int = 5) -> dict:
    if not aspects or not isinstance(text, str) or text.strip() == "":
        return {}

    doc = nlp(text)
    tokens = [token.text.lower() for token in doc]
    aspect_contexts = {}

    for aspect in aspects:
        aspect_tokens = aspect.split()
        aspect_len = len(aspect_tokens)

        for i in range(len(tokens) - aspect_len + 1):
            if tokens[i:i + aspect_len] == aspect_tokens:
                start = max(0, i - window_size)
                end = min(len(tokens), i + aspect_len + window_size)
                context = " ".join(tokens[start:end])

                if aspect in aspect_contexts:
                    aspect_contexts[aspect].append(context)
                else:
                    aspect_contexts[aspect] = [context]

    return aspect_contexts


def analyze_aspect_sentiment(aspect_contexts: dict) -> dict:
    sia = SentimentIntensityAnalyzer()
    aspect_sentiments = {}

    for aspect, contexts in aspect_contexts.items():
        sentiment_scores = [sia.polarity_scores(context)["compound"] for context in contexts]
        if sentiment_scores:
            aspect_sentiments[aspect] = sum(sentiment_scores) / len(sentiment_scores)

    return aspect_sentiments


def get_dominant_sentiment(aspect_sentiments: dict, threshold: float = 0.4) -> str:
    if not aspect_sentiments:
        return "Neutral"

    strong_positive = sum(1 for score in aspect_sentiments.values() if score >= threshold)
    strong_negative = sum(1 for score in aspect_sentiments.values() if score <= -threshold)
    avg_sentiment = sum(aspect_sentiments.values()) / len(aspect_sentiments)

    if strong_positive > strong_negative * 1.5:
        return "Positive"
    elif strong_negative > strong_positive * 1.5:
        return "Negative"
    elif abs(avg_sentiment) < 0.15:
        return "Neutral"
    elif avg_sentiment > 0:
        return "Positive"
    else:
        return "Negative"


def identify_contrast_markers(text: str) -> bool:
    contrast_patterns = [
        r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b",
        r"\byet\b", r"\bstill\b", r"\bdespite\b",
        r"\beven though\b", r"\bon the other hand\b", r"\bnevertheless\b"
    ]
    return any(re.search(pattern, text.lower()) for pattern in contrast_patterns)


def analyze_mixed_sentiment(text: str, aspect_sentiments: dict):
    if not aspect_sentiments:
        return False, "Neutral", None

    has_contrast = identify_contrast_markers(text)

    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for score in aspect_sentiments.values():
        sentiment_counts[classify_aspect_sentiment(score)] += 1

    has_positive = sentiment_counts["Positive"] > 0
    has_negative = sentiment_counts["Negative"] > 0

    is_mixed = (
        (has_contrast and has_positive and has_negative) or
        (has_positive and has_negative and min(sentiment_counts["Positive"], sentiment_counts["Negative"]) >= max(1, len(aspect_sentiments) // 3))
    )

    dominant = get_dominant_sentiment(aspect_sentiments)
    secondary = None

    if is_mixed:
        sentiments = ["Positive", "Neutral", "Negative"]
        sorted_sentiments = sorted(
            [(s, sentiment_counts[s]) for s in sentiments if sentiment_counts[s] > 0],
            key=lambda x: x[1],
            reverse=True
        )
        dominant = sorted_sentiments[0][0]
        if len(sorted_sentiments) > 1:
            secondary = sorted_sentiments[1][0]

    return is_mixed, dominant, secondary


def detect_solution_contexts(text: str):
    solution_phrases = [
        r"\bfix(ed)?\b", r"\bsolv(ed|e)?\b", r"\bsolution\b",
        r"\bwork\s?around\b", r"\buse\s+instead\b", r"\breplac(e|ed)\b",
        r"\balternative\b", r"\bsuggestion\b", r"\btip\b", r"\bhack\b"
    ]

    solution_segments = []
    for phrase in solution_phrases:
        for match in re.finditer(phrase, text.lower()):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            solution_segments.append(text[start:end])

    if not solution_segments:
        return False, None

    sia = SentimentIntensityAnalyzer()
    solution_sentiments = [sia.polarity_scores(segment)["compound"] for segment in solution_segments]
    avg_sentiment = sum(solution_sentiments) / len(solution_sentiments)

    return True, classify_aspect_sentiment(avg_sentiment)


def analyze_review_sentiment(text: str, rating: int, nlp, stop_words_set):
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral", [], {}, False, False, None

    if len(text.split()) < 5:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        final_sentiment = classify_aspect_sentiment(polarity)
        rating_implied = "Positive" if rating >= 4 else ("Negative" if rating <= 2 else "Neutral")
        misalignment = final_sentiment != rating_implied and abs(rating - 3) > 1
        return final_sentiment, [], {}, False, misalignment, None

    aspects = extract_aspects(text, nlp, stop_words_set)
    aspect_contexts = extract_aspect_contexts(text, aspects, nlp)
    aspect_sentiments = analyze_aspect_sentiment(aspect_contexts)

    is_mixed, dominant_sentiment, _ = analyze_mixed_sentiment(text, aspect_sentiments)
    has_solution, solution_sentiment = detect_solution_contexts(text)

    rating_implied_sentiment = "Positive" if rating >= 4 else ("Negative" if rating <= 2 else "Neutral")

    final_sentiment = dominant_sentiment
    misalignment = False

    if has_solution and solution_sentiment == "Positive" and rating <= 2:
        final_sentiment = "Negative"
        misalignment = True
    elif is_mixed and dominant_sentiment != rating_implied_sentiment:
        final_sentiment = rating_implied_sentiment
        misalignment = True
    elif dominant_sentiment != rating_implied_sentiment and abs(rating - 3) >= 2:
        final_sentiment = rating_implied_sentiment
        misalignment = True
    elif dominant_sentiment != rating_implied_sentiment and abs(rating - 3) == 1:
        misalignment = True
    elif dominant_sentiment == "Neutral" and rating_implied_sentiment != "Neutral":
        final_sentiment = rating_implied_sentiment
        misalignment = True
    else:
        misalignment = final_sentiment != rating_implied_sentiment

    return final_sentiment, aspects, aspect_sentiments, is_mixed, misalignment, solution_sentiment


def adjust_rating_based_on_sentiment(row):
    original_rating = row["rating"]
    absa_sentiment = row["absa_sentiment"]

    sentiment_bias = {"Positive": 1, "Neutral": 0, "Negative": -1}
    absa_bias = sentiment_bias.get(absa_sentiment, 0)

    rating_bias = 0
    if original_rating >= 4:
        rating_bias = 1
    elif original_rating <= 2:
        rating_bias = -1

    if absa_bias != 0 and absa_bias == rating_bias:
        return original_rating

    adjustment = 0
    if original_rating == 5 and absa_bias == -1:
        adjustment = -2
    elif original_rating == 1 and absa_bias == 1:
        adjustment = 2
    elif original_rating == 4 and absa_bias == -1:
        adjustment = -2
    elif original_rating == 2 and absa_bias == 1:
        adjustment = 2
    elif original_rating >= 4 and absa_bias == 0:
        adjustment = -1
    elif original_rating <= 2 and absa_bias == 0:
        adjustment = 1
    elif original_rating == 3 and absa_bias == 1:
        adjustment = 1
    elif original_rating == 3 and absa_bias == -1:
        adjustment = -1

    adjusted_rating = original_rating + adjustment
    return max(1, min(5, adjusted_rating))


def run_absa_analysis(
    original_data_path: str = "data/subset_data.csv",
    preprocessed_data_path: str = "data/preprocessed_data.csv",
    output_path: str = "data/absa_results.csv"
):
    nltk.download("stopwords")
    nltk.download("vader_lexicon")

    nlp = load_spacy_model()
    stop_words_set = set(stopwords.words("english"))

    original_df = pd.read_csv(original_data_path).fillna("")
    preprocessed_df = pd.read_csv(preprocessed_data_path).fillna("")

    original_for_merge = original_df[["rating", "title", "text"]].copy()
    original_for_merge["full_text_orig"] = original_for_merge[["title", "text"]].agg(" ".join, axis=1)
    original_for_merge["cleaned_text_for_merge"] = original_for_merge["full_text_orig"].apply(preprocess_text)
    original_for_merge.drop(columns=["title", "text", "full_text_orig"], inplace=True)
    original_for_merge.drop_duplicates(subset=["cleaned_text_for_merge"], keep="first", inplace=True)

    merged_df = pd.merge(
        preprocessed_df,
        original_for_merge,
        left_on="cleaned_text",
        right_on="cleaned_text_for_merge",
        how="inner"
    )

    analysis_df = merged_df.drop(columns=["cleaned_text_for_merge"]).copy()

    if "sentiment" not in analysis_df.columns:
        analysis_df["sentiment"] = analysis_df["rating"].apply(
            lambda x: "Positive" if x >= 4 else ("Neutral" if x == 3 else "Negative")
        )

    analysis_df.rename(columns={"sentiment": "original_sentiment"}, inplace=True)

    results = []

    for _, row in analysis_df.iterrows():
        text = row["cleaned_text"]
        rating = row["rating"]
        original_sentiment = row["original_sentiment"]

        final_sentiment, aspects, aspect_sentiments, is_mixed, misalignment, solution_sentiment = analyze_review_sentiment(
            text, rating, nlp, stop_words_set
        )

        aspect_sentiment_labels = {
            aspect: classify_aspect_sentiment(score)
            for aspect, score in aspect_sentiments.items()
        }

        results.append({
            "rating": rating,
            "cleaned_text": text,
            "original_sentiment": original_sentiment,
            "absa_sentiment": final_sentiment,
            "aspects": ", ".join(aspects) if aspects else "",
            "aspect_sentiments": str(aspect_sentiment_labels) if aspect_sentiment_labels else "{}",
            "is_mixed": bool(is_mixed),
            "has_misalignment": bool(misalignment),
            "solution_sentiment": solution_sentiment if solution_sentiment else "None"
        })

    results_df = pd.DataFrame(results)
    results_df["adjusted_rating"] = results_df.apply(adjust_rating_based_on_sentiment, axis=1)
    results_df["rating_adjustment"] = results_df["adjusted_rating"] - results_df["rating"]
    results_df["sentiment_match"] = results_df["original_sentiment"] == results_df["absa_sentiment"]

    results_df.to_csv(output_path, index=False)
    print(f"Saved ABSA results to: {output_path}")

    return results_df


if __name__ == "__main__":
    run_absa_analysis()
