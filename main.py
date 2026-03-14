from sklearn.model_selection import train_test_split

from src.data import load_data, basic_cleaning
from src.features import feature_engineering, preprocess
from src.models import get_models
from src.evaluate import evaluate_model
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import time


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
        start_time = time.time()        
        cv_results = cross_validate_model(model, X_train, y_train, k=5)
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        train_time = time.time() - start_time

        all_results[name] = {**results, **cv_results, "train_time": train_time}

        print(f"\n{name}")
        print(f"Training Time: {train_time:.2f}s")
        print(f"CV RMSE Mean: {cv_results['CV_RMSE_mean']:.4f}")
        print(f"CV RMSE Std: {cv_results['CV_RMSE_std']:.4f}")

        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    
    plot_performance_vs_time(all_results)
    best_model_name = min(all_results, key=lambda x: all_results[x]["CV_RMSE_mean"])
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    plot_predicted_vs_actual(best_model, X_test, y_test, best_model_name)
    plot_residuals(best_model, X_test, y_test, best_model_name)
    

def plot_performance_vs_time(all_results):
    model_names = list(all_results.keys())

    rmse_scores = [r["CV_RMSE_mean"] for r in all_results.values()]
    train_times = [r["train_time"] for r in all_results.values()]

    fig, ax1 = plt.subplots(figsize=(9,5))

    bars = ax1.bar(model_names, rmse_scores, color="skyblue", label="CV RMSE")
    ax1.set_ylabel("CV RMSE", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.2f}",
             ha="center", va="bottom")
    ax2 = ax1.twinx()
    ax2.plot(model_names, train_times, color="red", marker="o", label="Training Time")
    ax2.set_ylabel("Training Time (seconds)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    plt.title("Model Performance vs Training Time")
    ax1.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1))
    plt.show()

def plot_predicted_vs_actual(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    plt.figure(figsize=(6,6))

    plt.scatter(y_test, preds, alpha=0.5)

    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())

    plt.plot([min_val, max_val], [min_val, max_val], color="red")

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Predicted vs Actual ({model_name})")

    plt.tight_layout()
    plt.show()

def plot_residuals(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    residuals = y_test - preds

    plt.figure(figsize=(8,5))

    plt.hist(residuals, bins=50)

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution ({model_name})")

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

''' Latest results
Ridge
Training Time: 1.88s
CV RMSE Mean: 10125.2034
CV RMSE Std: 128.7503
MSE: 100705300.2353
RMSE: 10035.2030
MAE: 6959.1805
R2: 0.4814

Decision Tree
Training Time: 15.86s
CV RMSE Mean: 6519.8939
CV RMSE Std: 81.0203
MSE: 39592290.8362
RMSE: 6292.2405
MAE: 3108.9266
R2: 0.7961

Random Forest
Training Time: 775.35s
CV RMSE Mean: 5056.4718
CV RMSE Std: 81.2347
MSE: 24183928.5569
RMSE: 4917.7158
MAE: 2249.3165
R2: 0.8755

Neural Network
Training Time: 1130.56s
CV RMSE Mean: 7434.3828
CV RMSE Std: 315.2763
MSE: 54767324.8205
RMSE: 7400.4949
MAE: 4549.8320
R2: 0.7179
'''