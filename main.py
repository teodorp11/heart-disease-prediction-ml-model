from src.preprocess import clean_data, get_features_and_target
from src.model import train_model, evaluate_model
from src.visualize import plot_confusion_matrix


# 1. Clean Data
data = clean_data('data/framingham.csv')

# 2. Prepare Features
X, y = get_features_and_target(data)

# 3. Train
model, X_test, y_test = train_model(X, y)

# 4. Evaluate
acc, report, cm = evaluate_model(model, X_test, y_test)

print(f"Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

plot_confusion_matrix(cm)