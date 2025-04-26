import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Load the dataset
df = pd.read_csv("diabetes_dataset.csv")

# Create Diabetes target label
df['Diabetes'] = ((df['HbA1c'] >= 6.5) | (df['Fasting_Blood_Glucose'] >= 126)).astype(int)

# Drop unnecessary column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Encode categorical columns
categorical_cols = ['Sex', 'Ethnicity', 'Physical_Activity_Level', 'Alcohol_Consumption', 'Smoking_Status']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
x = df.drop(columns=['Diabetes'])
y = df['Diabetes']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# Define TabNet model
clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=1
)

# Convert to numpy
X_train_np, y_train_np = x_train.values, y_train.values
X_test_np, y_test_np = x_test.values, y_test.values

# Fit model
os.makedirs("results", exist_ok=True)
# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Train the model
clf.fit(
    X_train_np, y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_name=["test"],
    eval_metric=["accuracy"],
    max_epochs=35,
    patience=20,
    batch_size=512,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Predict
preds = clf.predict(X_test_np)
preds_proba = clf.predict_proba(X_test_np)[:, 1]

# Metrics
report = classification_report(y_test_np, preds)
conf_matrix = confusion_matrix(y_test_np, preds)
roc_auc = roc_auc_score(y_test_np, preds_proba)

# Print for debug
print("🧾 Classification Report:")
print(report)
print("📊 Confusion Matrix:")
print(conf_matrix)
print("📈 ROC-AUC Score:", roc_auc)

# Save results
try:
    with open("results/result_analysis.txt", "w") as f:
        f.write("Classification Report:\n" + report)
        f.write("\nConfusion Matrix:\n" + str(conf_matrix))
        f.write("\n\nROC-AUC Score: " + str(roc_auc))
    print(" Results saved to results/result_analysis.txt")
except Exception as e:
    print(" Error saving file:", e)
