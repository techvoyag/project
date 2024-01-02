import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import joblib

def get_best_run(experiment_id, metric_name, smaller_is_better=True):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if smaller_is_better else 'DESC'}"]
    )
    if not runs:
        raise ValueError("No runs found for the experiment")
    best_run = runs[0]
    return best_run

# Load data and preprocess
data = pd.read_csv('data/dummy_sensor_data.csv')
# Feature Engineering 
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month

# Select features and target
X = data[['Hour', 'Day', 'Month', 'Machine_ID', 'Sensor_ID']]
y = data['Reading']
categorical_cols = ['Machine_ID', 'Sensor_ID']
numeric_cols = ['Hour', 'Day', 'Month']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])
X_processed = preprocessor.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Load the best model from MLflow
experiment_id = "0"
metric_name = "mse"
best_run = get_best_run(experiment_id, metric_name, smaller_is_better=True)
best_run_id = best_run.info.run_id
model_path = f"app/models/0/{best_run_id}/artifacts/best_model"  # Adjust the path as needed
model = mlflow.sklearn.load_model(model_path)

# Evaluate the model
predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
print(f"Loaded model MSE: {mse}")

# If MSE is greater than 500, retrain the model
if mse > 500:
    print("Retraining the model...")
    with mlflow.start_run():
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=2, n_jobs=2, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Log parameters, metrics, and model
        mlflow.log_params(grid_search.best_params_)
        retrained_mse = mean_squared_error(y_val, best_model.predict(X_val))
        mlflow.log_metric("mse", retrained_mse)
        mlflow.sklearn.log_model(best_model, "retrained_model")

        # Save the retrained model
        model_filename = 'model/retrained_random_forest_model.joblib'
        joblib.dump(best_model, model_filename)
        print(f"Retrained model saved as {model_filename}")
else:
    print("No retraining needed.")
