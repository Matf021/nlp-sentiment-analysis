import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def line_graph(data_source, title, xlabel, ylabel, is_rotated=True, height=8, width=14):
    plt.figure(figsize=(width, height))

    if isinstance(data_source, pd.Series):
        ax = data_source.plot(kind="line", marker="o", color="skyblue", linewidth=2)
        for x, y in zip(data_source.index, data_source.values):
            plt.text(x, y, f"{y:.0f}", ha="center", va="bottom")
    elif isinstance(data_source, pd.DataFrame):
        data_source.plot(kind="line", marker="o", figsize=(width, height))
    else:
        raise ValueError("data_source must be a pandas Series or DataFrame.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if is_rotated:
        plt.xticks(rotation=45, ha="right")

    plt.legend(loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def bar_chart(data_source, title, xlabel, ylabel, is_rotated=True, height=14, width=7):
    plt.figure(figsize=(height, width))

    if isinstance(data_source, pd.Series):
        ax = data_source.plot(kind="bar", color="skyblue", edgecolor="black")
        for bar in ax.patches:
            ax.annotate(
                f"{bar.get_height():.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom"
            )
    elif isinstance(data_source, pd.DataFrame):
        data_source.plot(kind="bar", figsize=(height, width))
    else:
        raise ValueError("data_source must be a pandas Series or DataFrame.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if is_rotated:
        plt.xticks(rotation=45, ha="right")

    plt.show()


def run_eda(df: pd.DataFrame) -> None:
    print("Dataset information:")
    print(df.head())
    print(df.info())

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Time stamp data type: {df['timestamp'].dtype}")
    print("Detailed Information of rating:")
    print(df["rating"].describe())
    print("Checking missing value for each column:")
    print(df.isnull().sum())

    non_zero_helpful = df[df["helpful_vote"] > 0]
    print("Checking helpful_vote column:")
    print(non_zero_helpful["helpful_vote"].describe())

    plt.figure(figsize=(8, 5))
    avg_votes = df.groupby("rating")["helpful_vote"].mean().reset_index()
    sns.barplot(data=avg_votes, x="rating", y="helpful_vote")
    plt.title("Average Helpful Votes per Rating")
    plt.xlabel("Rating")
    plt.ylabel("Average Helpful Votes")
    plt.show()

    df["review_word_count"] = df["text"].str.len()
    print("Checking helpful_vote correlated to text length:")
    print(df[["helpful_vote", "review_word_count"]].corr())

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="review_word_count", y="helpful_vote")
    plt.title("Review Length vs. Helpful Votes")
    plt.xlabel("Review Length (characters)")
    plt.ylabel("Helpful Votes")
    plt.show()

    rating_distribution = df["rating"].value_counts().sort_index()
    bar_chart(rating_distribution, "Distribution of the number of review across products", "Rating", "Count", False, 4, 6)

    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    monthly_review_counts = df.groupby("year_month").size().reset_index(name="review_count")
    monthly_review_counts["year_month"] = pd.to_datetime(monthly_review_counts["year_month"], format="%Y-%m")

    line_graph(
        data_source=monthly_review_counts.set_index("year_month")["review_count"],
        title="Amazon Appliance Reviews Over Time",
        xlabel="Year-Month",
        ylabel="Number of Reviews"
    )
