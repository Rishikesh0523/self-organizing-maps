import os
import numpy as np
import pandas as pd
import kagglehub  # Ensure you have this module installed

# --- Download the latest version of the customer dataset from Haggle (Kaggle) ---
path = kagglehub.dataset_download("shrutimechlearn/customer-data")
# --- Load the dataset using pandas ---
# Assuming the downloaded file is named 'customer_data.csv' inside the given path.
csv_file = os.path.join(path, "Mall_Customers.csv")
customer_df = pd.read_csv(csv_file)
# Convert the DataFrame to a NumPy array for the SOM (selecting 'Age' and 'Income' columns)
data = customer_df[['Age', 'Annual_Income_(k$)']].values
# print data size
print(f"Data size: {data.shape}")
