from src.eda import run_eda
from src.preprocessing import save_preprocessed_data
from src.modeling import run_modeling_pipeline
from src.lexicon import run_lexicon_pipeline
from src.absa import run_absa_analysis


def main():
    run_eda("data/subset_data.csv")
    save_preprocessed_data("data/subset_data.csv")
    run_modeling_pipeline("data/remaining_reviews.csv")
    run_lexicon_pipeline("data/lexicon_reviews.csv")
    run_absa_analysis("data/subset_data.csv", "data/preprocessed_data.csv")


if __name__ == "__main__":
    main()
