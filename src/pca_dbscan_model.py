import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
import joblib

# Step 1: Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    df_clean = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df_pivot = df_clean.pivot_table(index='Customer_ID', columns='Timestamp', values='Consumption_kWh', aggfunc='sum')
    df_pivot.fillna(0, inplace=True)
    return df_pivot

# Step 2: Build the PCA + DBSCAN Pipeline
def build_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('dbscan', DBSCAN(eps=1.2, min_samples=5))
    ])
    return pipeline

# Step 3: Train the Model
def train_model(data, pipeline):
    pipeline.fit(data)
    cluster_labels = pipeline.named_steps['dbscan'].fit_predict(
        pipeline.named_steps['pca'].transform(
            pipeline.named_steps['scaler'].transform(data)
        )
    )
    data['Cluster'] = cluster_labels
    return pipeline, data

# Step 4: Save the Model
def save_model(pipeline, filename='pca_dbscan_model.pkl'):
    joblib.dump(pipeline, filename)

if __name__ == "__main__":
    file_path = 'cleaned_electricity_data.csv'
    df_pivot = load_and_preprocess_data(file_path)
    pipeline = build_pipeline()
    trained_pipeline, clustered_data = train_model(df_pivot, pipeline)
    save_model(trained_pipeline)
    print("Model trained and saved successfully!")
