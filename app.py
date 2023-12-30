from flask import Flask, jsonify
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Function to get the best run based on a metric
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

# Load the model from the best run
experiment_id = "0"  # Replace with your experiment ID
metric_name = "mse"  # Replace with your metric name

best_run = get_best_run(experiment_id, metric_name, smaller_is_better=True)
best_run_id = best_run.info.run_id
#model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/best_model")
# Assuming the model is in the 'models' subdirectory within the mounted volume
model_path = f"/app/models/0/{best_run_id}/artifacts/best_model"
model = mlflow.sklearn.load_model(model_path)

# Sample data for prediction
# Sample data for prediction
sample_data = {
    "Timestamp": ["2023-01-01 00:00:00"] * 15,  # 5 machines * 3 sensors
    "Machine_ID": ["Machine_1", "Machine_2", "Machine_3", "Machine_4", "Machine_5"] * 3,
    "Sensor_ID": ["Sensor_1", "Sensor_2", "Sensor_3"] * 5,
    "Reading": [109.93, 110.12, 108.56, 111.34, 109.78,  # Sample readings for each machine-sensor combination
                110.45, 109.96, 107.89, 112.00, 108.67,
                111.23, 109.50, 108.34, 113.21, 109.99]
}





@app.route('/predict', methods=['GET'])
def predict():
    try:
        sample_df = pd.DataFrame(sample_data)
        sample_df['Timestamp'] = pd.to_datetime(sample_df['Timestamp'])  # Corrected line
        sample_df['Hour'] = sample_df['Timestamp'].dt.hour
        sample_df['Day'] = sample_df['Timestamp'].dt.day
        sample_df['Month'] = sample_df['Timestamp'].dt.month
        # Select features for prediction
        features = sample_df[['Hour', 'Day', 'Month', 'Machine_ID', 'Sensor_ID']]
        categorical_cols = ['Machine_ID', 'Sensor_ID']
        numeric_cols = ['Hour', 'Day', 'Month']

        # Create a column transformer with OneHotEncoder for categorical columns and StandardScaler for numeric columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(), categorical_cols)
            ])
        
        X =  features[['Hour', 'Day', 'Month', 'Machine_ID', 'Sensor_ID']]
        # Fit and transform the features
        X_processed = preprocessor.fit_transform(X)
        # Predict
        prediction = model.predict(X_processed)

        # Print prediction
        print("Prediction:", prediction)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

