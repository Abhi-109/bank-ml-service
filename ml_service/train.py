import pandas as pd
import joblib  # Using joblib for model persistence (often better for scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

print("Training script started...")

# --- 1. Load Data ---
try:
    # We're running this script from the root, so the path is relative to the root
    # Or, to be safer, let's make paths relative to this file
    # (Though for Docker, running from root is common)
    # For simplicity now, let's assume we run from the root: `python ml_service/train.py`
    df = pd.read_csv('data/bank-full.csv', sep=';')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/bank.csv' not found. Make sure you're running this script from the project root folder.")
    exit()

# --- 2. Define Features and Target ---
# From our EDA on Day 2
numeric_features = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'
]

categorical_features = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'
]

# Map target variable 'y'
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Define X and y
X = df[numeric_features + categorical_features]
y = df['y']

print(f"Features ({len(numeric_features + categorical_features)}): {numeric_features + categorical_features}")
print(f"Target: 'y'")

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- 4. Create Preprocessing Pipeline ---
# Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # Ignores new categories at prediction time

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any other columns (though we don't have any)
)
print("ColumnTransformer created.")

# --- 5. Create Full Model Pipeline ---
# This is our baseline Logistic Regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced')) # Baseline model
])
print("Model Pipeline created.")

# --- 6. Train Model ---
print("Training the baseline model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 7. Evaluate Model ---
print("Evaluating model on test set...")
y_pred = model.predict(X_test)

# We use 'pos_label=1' because 'yes' is now 1
# This is the F1 score for the positive class ('yes')
f1 = f1_score(y_test, y_pred, pos_label=1) 
print(f"\n--- Baseline Model F1 Score (for 'yes' class): {f1:.4f} ---")

print("\n--- Classification Report ---")
# '1' corresponds to 'yes'
print(classification_report(y_test, y_pred, target_names=['no (0)', 'yes (1)']))

# --- 8. Save Model ---
model_path = 'ml_service/model.pkl'
joblib.dump(model, model_path)
print(f"\nModel saved successfully to {model_path}")

print("\nTraining script finished.")