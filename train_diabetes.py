from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
import os

# ------------------------
# Data Loading and Preprocessing
# ------------------------
def load_and_preprocess_data(csv_path):
    # Charger les données
    df = pd.read_csv(csv_path)

    # Split Train/Test (80% training, 20% test)
    X = df.drop("Outcome", axis=1).values  # Features
    y = df["Outcome"].values  # Target variable

    # Split the data into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler parameters to JSON
    scaler_params = {
        "min": scaler.data_min_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    if not os.path.exists("MLModels"):
        os.makedirs("MLModels")
    with open("MLModels/scaler_params.json", "w") as f:
        json.dump(scaler_params, f)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ------------------------
# Train XGBoost Model with Hyperparameter Tuning
# ------------------------
def train_model(X_train, y_train, X_test, y_test):
    # Initialize XGBoost model (similar to FastTree)
    model = xgb.XGBClassifier(
        n_estimators=300,  # Increase the number of boosting rounds
        learning_rate=0.1,  # Set learning rate
        max_depth=10,  # Depth of each tree (increase depth)
        objective="binary:logistic",  # Binary classification
        eval_metric="logloss",  # Log loss as evaluation metric
        subsample=0.8,  # Subsample ratio of the training instances
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        gamma=0.1,  # Regularization term
        seed=42  # Fix random seed for reproducibility
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (diabetes)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"📉 Model Accuracy: {accuracy * 100:.2f}%")
    print(f"📊 AUC Score: {auc_score:.4f}")

    # Save the model to disk
    model_path = "MLModels/diabetes_model.xgb"
    model.save_model(model_path)

    return model_path, accuracy, auc_score, model

# ------------------------
# Export Model to ONNX
# ------------------------
def export_to_onnx(model, scaler, onnx_model_path="MLModels/diabetes_model.onnx"):
    # Specify the input type (number of features: 8, type: float32)
    initial_type = [('input', FloatTensorType([None, 8]))]  # 8 features in input
    
    # Export the trained model to ONNX format using onnxmltools
    model_onnx = onnxmltools.convert.convert_xgboost(model, initial_types=initial_type)
    
    # Save the ONNX model to the specified file
    onnxmltools.utils.save_model(model_onnx, onnx_model_path)
    print(f"📦 Model exported to ONNX: {onnx_model_path}")

# ------------------------
# Main Entry
# ------------------------
def main():
    # Define file paths
    model_path = "MLModels/diabetes_model.xgb"
    scaler_path = "MLModels/scaler_params.json"
    onnx_model_path = "MLModels/diabetes_model.onnx"

    # Load the trained model and scaler
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("diabetes.csv")

    # Train the model
    model_path, accuracy, auc_score, model = train_model(X_train, y_train, X_test, y_test)

    # Export the trained model to ONNX
    export_to_onnx(model, scaler, onnx_model_path)

if __name__ == "__main__":
    main()
