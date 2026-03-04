import pandas as pd


def feature_engineering(df):
    current_year = 2026
    df["vehicle_age"] = current_year - df["year"]
    df["miles_per_year"] = df["odometer"] / df["vehicle_age"].replace(0, 1)

    if "cylinders" in df.columns:
        df["cylinders"] = (
            df["cylinders"]
            .str.extract(r"(\d+)")
            .astype(float)
        )
    df = df.drop(columns=["posting_date"], errors="ignore")
    return df


def preprocess(df):
    categorical_cols = [
        "manufacturer",
        "fuel",
        "transmission",
        "condition",
        "title_status",
        "drive",
        "size",
        "type",
        "paint_color"
    ]

    categorical_cols = categorical_cols = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.astype(float)

    return df