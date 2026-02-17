from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model(X, y):
    """
    Splits data into 70/30 sets and trains a Logistic Regression model.lear
    
    Returns:
        tuple: (fitted_model, X_test, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4
    )
    model = LogisticRegression().fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Predicts outcomes and evaluates performance using accuracy, 
    precision/recall reports, and a confusion matrix.
    
    Returns:
        tuple: (float: accuracy, str: report, ndarray: conf_matrix)
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, conf_matrix