import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json  # <-- for saving scaler params

# ------------------------
# Neural Network Definition
# ------------------------
class DiabetesNet(nn.Module):
    def __init__(self):
        super(DiabetesNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ------------------------
# Data Loading and Processing
# ------------------------
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler params for later use in .NET
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open("scaler_params.json", "w") as f:
        json.dump(scaler_params, f)
    print("✅ Scaler parameters saved to scaler_params.json")

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------------
# Training
# ------------------------
def train_model(X_train, y_train, epochs=100, lr=0.001):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = DiabetesNet()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train_t)
        loss = loss_fn(y_pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model

# ------------------------
# ONNX Export (with input: "input")
# ------------------------
def export_model_to_onnx(model, filepath="diabetes_model.onnx"):
    dummy_input = torch.randn(1, 8)  # Shape = [batch_size, 8 features]

    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )

    print(f"✅ Model exported to ONNX: {filepath}")

# ------------------------
# Main Script
# ------------------------
def main():
    print("📥 Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data("diabetes.csv")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("🏋️‍♂️ Training model...")
    model = train_model(X_train, y_train)

    print("💾 Saving model weights...")
    torch.save(model.state_dict(), "diabetes_model.pth")

    print("📦 Exporting to ONNX...")
    export_model_to_onnx(model)

if __name__ == "__main__":
    main()
