# Machine Learning Project with Concept Drift Monitoring


## Features
- **Model Training**: Trains a RandomForest model using sensor data.
- **Automated Retraining**: Automatically retrains the model when new data is added or a push is made on the repo, ensuring the model remains accurate over time.

## Getting Started
These instructions will help you set up the project and run it on your local machine for development and testing purposes.

### Prerequisites
- Python 3.8+
- Required Python libraries: 
    `pandas`
    `scikit-learn`
    `mlflow`
    `dvc`
    `dvc_gdrive`
    `joblib`

### Prerequisites
- the docker image on dockerhub/techovyag can also me used
### Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/techvoyag/project.git
cd project
pip install -r requirements.txt
