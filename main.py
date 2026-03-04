from sklearn.model_selection import train_test_split
import numpy as np

from src.data import load_data, basic_cleaning
from src.features import feature_engineering, preprocess
from src.models import get_models
from src.evaluate import evaluate_model
import matplotlib.pyplot as plt


def main():
    df = load_data("data/vehicles.csv")

    df = basic_cleaning(df)

    df = feature_engineering(df)

    y = np.log1p(df["price"])
    X = df.drop(columns=["price"])
    

    X = preprocess(X)
    
    X = X.dropna()  
    y = y.loc[X.index]

    non_numeric = X.select_dtypes(exclude=["number"]).columns
    if len(non_numeric) > 0:
        print("Dropping non-numeric columns:", non_numeric)
        X = X.drop(columns=non_numeric)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models()

    all_results = {}

    for name, model in models.items():
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        all_results[name] = results

        print(f"\n{name}")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    model_names = list(all_results.keys())
    rmse_scores = [results["RMSE"] for results in all_results.values()]
    
    plt.bar(model_names, rmse_scores)
    plt.xticks(rotation=45)
    plt.ylabel("RMSE")
    plt.title("Model Comparison")
    plt.show()


if __name__ == "__main__":
    main()

'''Initial results
Ridge
MSE: 0.6793
RMSE: 0.8242
MAE: 0.5356
R2: 0.3590

Random Forest
MSE: 0.2987
RMSE: 0.5466
MAE: 0.2422
R2: 0.7181

Neural Network
MSE: 0.7813
RMSE: 0.8839
MAE: 0.5657
R2: 0.2627
'''