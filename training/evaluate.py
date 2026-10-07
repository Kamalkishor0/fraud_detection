from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, PrecisionRecallDisplay
)

def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    avg_precision = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nModel: {model_name}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1:                {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    disp = PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test
    )
    disp.ax_.set_title(
        f"Precision-Recall Curve - {model_name}"
    )

    return {
        "model": model_name,
        "average_precision": avg_precision,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }