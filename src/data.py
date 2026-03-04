import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def basic_cleaning(df):
    df = df[(df["price"] > 100) & (df["price"] < 100000)]

    key_cols = ["price", "year", "odometer", "manufacturer", "fuel", "transmission"]
    df = df.dropna(subset=key_cols)

    drop_cols = [
        "id",
        "url",
        "region",
        "region_url",
        "VIN",
        "image_url",
        "description",
        "county",
        "state",
        "lat",
        "long",
        "model"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return df