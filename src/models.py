from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


def get_models():
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=20, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
    }

    return models