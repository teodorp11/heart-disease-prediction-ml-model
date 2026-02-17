import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm):
    """
    Generates a heatmap of the confusion matrix using a 
    Green-Blue (Petrol) color scheme.
    
    Args:
        cm (ndarray): The confusion matrix from sklearn.metrics.
    """
    # Convert matrix to a labeled DataFrame for better plotting
    conf_matrix = pd.DataFrame(
        data=cm, 
        columns=['Predicted: No CHD', 'Predicted: CHD'], 
        index=['Actual: No CHD', 'Actual: CHD']
    )
    
    plt.figure(figsize=(8, 5))
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="GnBu", cbar=True)
    
    plt.title("Confusion Matrix for Heart Disease Prediction", fontsize=14, pad=20)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()