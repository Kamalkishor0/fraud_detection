import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_PATH = Path(os.environ["DATA_PATH"])
MODEL_PATH = Path(os.environ["MODEL_PATH"])
if not DATA_PATH.is_absolute():
    DATA_PATH = PROJECT_ROOT / DATA_PATH
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

# 1. Load
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

# 5. Train
pipeline.fit(X_train, y_train)

# 6. Save
joblib.dump(pipeline, MODEL_PATH)