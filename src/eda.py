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
    plt.tight_layout()
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

    plt.tight_layout()
    plt.show()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_data_info(df: pd.DataFrame) -> None:
    print("Dataset information:")
    print(df.head())
    print(df.info())

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Time stamp data type: {df['timestamp'].dtype}")

    print("Detailed Information of rating:")
    print(df["rating"].describe())

    print("Checking missing value for each column:")
    print(df.isnull().sum())


def helpful_vote_analysis(df: pd.DataFrame) -> None:
    non_zero_helpful = df[df["helpful_vote"] > 0]

    print("Checking helpful_vote column:")
    print(non_zero_helpful["helpful_vote"].describe())

    print("Checking helpful_vote distribution rating:")
    plt.figure(figsize=(8, 5))
    avg_votes = df.groupby("rating")["helpful_vote"].mean().reset_index()
    sns.barplot(data=avg_votes, x="rating", y="helpful_vote")
    plt.title("Average Helpful Votes per Rating")
    plt.xlabel("Rating")
    plt.ylabel("Average Helpful Votes")
    plt.tight_layout()
    plt.show()

    df["review_word_count"] = df["text"].str.len()
    print("Checking helpful_vote correlated to text length:")
    print(df[["helpful_vote", "review_word_count"]].corr())

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="review_word_count", y="helpful_vote")
    plt.title("Review Length vs. Helpful Votes")
    plt.xlabel("Review Length (characters)")
    plt.ylabel("Helpful Votes")
    plt.tight_layout()
    plt.show()


def unique_item_product_counts(df: pd.DataFrame) -> None:
    unique_asins_ver = df["asin"].unique()
    print(f"Number of unique items: {len(unique_asins_ver)}")

    unique_asins = df["parent_asin"].unique()
    print(f"Number of unique products: {len(unique_asins)}")


def product_average_rating_pie(df: pd.DataFrame) -> None:
    average_rating_per_product = df.groupby("parent_asin")["rating"].mean()
    rating_counts = average_rating_per_product.round().value_counts().sort_index()

    for rating in range(1, 6):
        if rating not in rating_counts:
            rating_counts[rating] = 0

    rating_counts = rating_counts.sort_index()

    plt.figure(figsize=(7, 7))
    colors = ["#F6BD60", "#F7EDE2", "#F5CAC3", "#84A59D", "#F28482"]
    explode = (0.1, 0.05, 0.05, 0.05, 0.1)

    plt.pie(
        rating_counts,
        labels=rating_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        explode=explode,
        shadow=True,
        startangle=140
    )

    plt.title("Percentage of Products by Average Rating")
    plt.tight_layout()
    plt.show()


def rating_distribution_analysis(df: pd.DataFrame) -> None:
    rating_distribution = df["rating"].value_counts().sort_index()
    bar_chart(
        rating_distribution,
        title="Distribution of the number of review across products",
        xlabel="Rating",
        ylabel="Count",
        is_rotated=False,
        height=4,
        width=6
    )


def monthly_review_trend(df: pd.DataFrame) -> None:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)

    monthly_review_counts = df.groupby("year_month").size().reset_index(name="review_count")
    monthly_review_counts["year_month"] = pd.to_datetime(
        monthly_review_counts["year_month"],
        format="%Y-%m"
    )

    line_graph(
        data_source=monthly_review_counts.set_index("year_month")["review_count"],
        title="Amazon Appliance Reviews Over Time",
        xlabel="Year-Month",
        ylabel="Number of Reviews"
    )


def item_review_distribution(df: pd.DataFrame) -> None:
    item_review_counts = df["asin"].value_counts()

    print("Review count across items:")
    print(item_review_counts.describe())

    percentiles = item_review_counts.quantile([0.75, 0.9, 0.95, 0.99, 1.0])
    print("Check the outliers (percentile):")
    print(percentiles)

    threshold = item_review_counts.quantile(0.99)
    outlier_asins = item_review_counts[item_review_counts > threshold].index
    outlier_items = df[df["asin"].isin(outlier_asins)]

    print("Get the statistic info from outliers:")
    print(outlier_items["rating"].describe())


def product_review_distribution(df: pd.DataFrame) -> None:
    product_review_counts = df["parent_asin"].value_counts()

    print("Review count across product:")
    print(product_review_counts.describe())

    percentiles = product_review_counts.quantile([0.75, 0.9, 0.95, 0.99, 1.0])
    print("Check the outliers (percentile):")
    print(percentiles)

    threshold = product_review_counts.quantile(0.99)
    outlier_asins = product_review_counts[product_review_counts > threshold].index
    outlier_products = df[df["parent_asin"].isin(outlier_asins)]

    print("Get the statistic info from outliers:")
    print(outlier_products["rating"].describe())

    filtered_counts = product_review_counts[product_review_counts > threshold]
    print(f"The products have more than {threshold} reviews:")
    print(filtered_counts)

    top_products = filtered_counts.index
    filtered_df = df[df["parent_asin"].isin(top_products)]

    rating_distribution = filtered_df.groupby(["parent_asin", "rating"]).size().unstack(fill_value=0)
    rating_distribution.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")

    plt.xlabel("Parent ASIN")
    plt.ylabel("Number of Reviews")
    plt.title("Rating Distribution for Most Reviewed Products")
    plt.xticks(rotation=90)
    plt.legend(title="Rating", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def user_review_analysis(df: pd.DataFrame) -> None:
    user_review_counts = df["user_id"].value_counts()

    print("\nDistribution of Reviews per User:")
    print(user_review_counts.describe())

    outliers = user_review_counts[user_review_counts > 1]
    print(f"\nUsers with more than 1 review:\n{outliers}")

    duplicate_reviews = df.duplicated(subset=["user_id", "parent_asin", "text"])
    print(f"\nNumber of duplicate reviews: {duplicate_reviews.sum()}")

    rating_5_df = df[df["rating"] == 5]
    rating_5_grouped = rating_5_df.groupby(["user_id", "parent_asin"]).size()
    multiple_5_for_product = rating_5_grouped[rating_5_grouped > 1]

    print("Users with multiple 5-star ratings for the same product:")
    print(multiple_5_for_product)

    duplicates = rating_5_df["text"][rating_5_df["text"].duplicated()]
    print("Duplicated Texts:")
    print(pd.DataFrame(duplicates))


def review_length_analysis(df: pd.DataFrame) -> None:
    df = df.copy()
    df["review_length"] = df["text"].str.len()

    print("\nReview Length Statistics:")
    print(df["review_length"].describe())

    threshold = df["review_length"].quantile(0.75)
    stats_above_threshold = df.loc[df["review_length"] > threshold, "review_length"].describe()

    print(f"The text review more than {threshold}:")
    print(stats_above_threshold)

    pd.set_option("display.max_colwidth", None)

    outlier_text = df.loc[df["review_length"] == df["review_length"].max(), "text"]
    print("The outlier text:")
    print(outlier_text)


def run_eda(path: str) -> pd.DataFrame:
    df = load_dataset(path)

    basic_data_info(df)
    helpful_vote_analysis(df)
    unique_item_product_counts(df)
    product_average_rating_pie(df)
    rating_distribution_analysis(df)
    monthly_review_trend(df)
    item_review_distribution(df)
    product_review_distribution(df)
    user_review_analysis(df)
    review_length_analysis(df)

    return df


if __name__ == "__main__":
    run_eda("data/subset_data.csv")
