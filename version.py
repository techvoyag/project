import sys

# Import the required packages
import pandas as pd
import sklearn
import mlflow

# Add 'dvc' and 'dvc_gdrive' to the list of packages, though they might not be available
packages = ["pandas", "sklearn", "mlflow", "dvc"]

# Function to get the version of a package
def get_package_version(package_name):
    try:
        return __import__(package_name).__version__
    except ImportError:
        return "not installed"

# Create a dictionary with package names and their versions
package_versions = {package: get_package_version(package) for package in packages}

# Print package versions
for package, version in package_versions.items():
    print(f"{package}: {version}")

# Write the requirements to a file
requirements = [f"{package}=={version}" for package, version in package_versions.items() if version != "not installed"]

# Write to a file named 'requirements.txt'
with open("requirements.txt", "w") as file:
    file.writelines("\n".join(requirements))

print("\nrequirements.txt file created.")
