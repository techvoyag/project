import mlflow

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

# Replace with your experiment ID
experiment_id = "0"  # Use your actual experiment ID
metric_name = "mse"  # The metric to compare

# Get the best run
best_run = get_best_run(experiment_id, metric_name, smaller_is_better=True)
best_run_id = best_run.info.run_id

# Load the model from the best run for prediction
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/best_model")
print("Best Run ID:", best_run_id)
# Now you can use this model for prediction
