import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Load
df = pd.read_csv("../data/creditcard.csv")

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
joblib.dump(pipeline,"../ml/artifacts/modellr.pkl")