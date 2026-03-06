import matplotlib.pyplot as plt
import seaborn as sns
from src.data import load_data, basic_cleaning
from src.features import feature_engineering

def dataset_statistics(df):
    print(f"Number of records: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")

    print("\nFeature types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nTarget variable statistics (price):")
    print(df["price"].describe())

def feature_summary(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        print(f"\nFeature: {col}")
        print(df[col].describe())

def plot_price_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df["price"], bins=50, kde=True)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.show()

def plot_age_vs_price(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="vehicle_age", y="price", data=df)
    plt.title("Vehicle Age vs Price")
    plt.show()  

def plot_mileage_vs_price(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="odometer", y="price", data=df)
    plt.title("Mileage vs Price")
    plt.show()

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def main():    
    df = load_data("data/vehicles.csv")
    df = basic_cleaning(df)
    df = feature_engineering(df)
    dataset_statistics(df)
    feature_summary(df)
    plot_price_distribution(df)
    plot_age_vs_price(df)
    plot_mileage_vs_price(df)
    correlation_heatmap(df)


if __name__ == "__main__":
    main()