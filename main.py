from sklearn.model_selection import train_test_split

from src.data import load_data, basic_cleaning
from src.features import feature_engineering, preprocess
from src.models import get_models
from src.evaluate import evaluate_model
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


def cross_validate_model(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=kf
    )

    rmse_scores = -scores
    return {
        "CV_RMSE_mean": np.mean(rmse_scores),
        "CV_RMSE_std": np.std(rmse_scores)
    }

def main():
    df = load_data("data/vehicles.csv")

    df = basic_cleaning(df)

    df = feature_engineering(df)

    y = df["price"]
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
        cv_results = cross_validate_model(model, X_train, y_train, k=5)
        results = evaluate_model(model, X_train, X_test, y_train, y_test)

        all_results[name] = {**results, **cv_results}

        print(f"\n{name}")

        print(f"CV RMSE Mean: {cv_results['CV_RMSE_mean']:.4f}")
        print(f"CV RMSE Std: {cv_results['CV_RMSE_std']:.4f}")

        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

    model_names = list(all_results.keys())
    rmse_scores = [results["CV_RMSE_mean"] for results in all_results.values()]
    
    plt.bar(model_names, rmse_scores)
    plt.xticks(rotation=45)
    plt.ylabel("RMSE")
    plt.title("Model Comparison")
    plt.show()


if __name__ == "__main__":
    main()

'''Initial results

Ridge
CV RMSE Mean: 10125.2034
CV RMSE Std: 128.7503
MSE: 100705300.2353
RMSE: 10035.2030
MAE: 6959.1805
R2: 0.4814

Decision Tree
CV RMSE Mean: 6519.8939
CV RMSE Std: 81.0203
MSE: 39592290.8362
RMSE: 6292.2405
MAE: 3108.9266
R2: 0.7961

Random Forest
CV RMSE Mean: 5056.4718
CV RMSE Std: 81.2347
MSE: 24183928.5569
RMSE: 4917.7158
MAE: 2249.3165
R2: 0.8755

Neural Network
CV RMSE Mean: 7563.7183
CV RMSE Std: 298.1698
MSE: 54615751.7988
RMSE: 7390.2471
MAE: 4585.3100
R2: 0.7187
'''