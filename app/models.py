import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_engagement_model(X, y, model_path="models/engagement_rf.pkl"):
    """
    Train a Random Forest model to predict engagement_score.
    Save it to model_path and return model + metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model, mae, r2

def load_engagement_model(model_path="models/engagement_rf.pkl"):
    """
    Load a previously saved model from disk.
    """
    return joblib.load(model_path)
