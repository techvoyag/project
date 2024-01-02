import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load data
data = pd.read_csv('data/dummy_sensor_data.csv')  # file path

# Feature Engineering (if needed)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month

# Select features and target
X = data[['Hour', 'Day', 'Month', 'Machine_ID', 'Sensor_ID']]
y = data['Reading']

# Define categorical columns and numeric columns
categorical_cols = ['Machine_ID', 'Sensor_ID']
numeric_cols = ['Hour', 'Day', 'Month']

# Create a column transformer with OneHotEncoder for categorical columns and StandardScaler for numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# Initialize the model

model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)

# Train the model
model.fit(X_train, y_train)

# Validate the model
predictions = model.predict(X_val)
mse = mean_squared_error(y_val, predictions)
print(f'Mean Squared Error: {mse}')


# Define a smaller parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

# Initialize Grid Search with fewer cross-validation folds and limited parallel jobs
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=2, verbose=2)

# Perform grid search
grid_search.fit(X_train, y_train)

# Start an MLflow run# After grid search is completed
with mlflow.start_run() as run:
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Log the best parameters and the best model
    mlflow.log_params(best_params)
    
    # Compute and log the best MSE
    best_mse = mean_squared_error(y_val, best_model.predict(X_val))
    mlflow.log_metric("best_mse", best_mse)

    # Log the best model
    #mlflow.sklearn.log_model(best_model, "best_model")
    # Log the best model
    #mlflow.sklearn.log_model(best_model, "model/best_model")
    # Save the best model using joblib
    
    model_filename = 'model/random_forest_model.joblib'
    joblib.dump(best_model, model_filename)


    # Print the run ID
    print("MLflow Run ID:", run.info.run_id)