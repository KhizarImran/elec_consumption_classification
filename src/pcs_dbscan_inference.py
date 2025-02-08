import pandas as pd
import joblib

# Step 1: Load the Trained Model
def load_model(model_path='pca_dbscan_model.pkl'):
    return joblib.load(model_path)

# Step 2: Load New Data for Prediction
def load_new_data(file_path):
    df_new = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df_pivot = df_new.pivot_table(index='Customer_ID', columns='Timestamp', values='Consumption_kWh', aggfunc='sum')
    df_pivot.fillna(0, inplace=True)
    return df_pivot

# Step 3: Predict Clusters for New Data
def predict_clusters(model, new_data):
    cluster_labels = model.named_steps['dbscan'].fit_predict(
        model.named_steps['pca'].transform(
            model.named_steps['scaler'].transform(new_data)
        )
    )
    new_data['Cluster'] = cluster_labels
    return new_data

if __name__ == "__main__":
    model = load_model('pca_dbscan_model.pkl')
    file_path = 'new_customer_data.csv'  # Replace with your new data file
    new_data = load_new_data(file_path)
    result = predict_clusters(model, new_data)
    
    print("Cluster Predictions:\n", result[['Cluster']].value_counts())
    
    # Optional: Save the results
    result.to_csv('predicted_clusters.csv')
    print("Cluster predictions saved to 'predicted_clusters.csv'!")
