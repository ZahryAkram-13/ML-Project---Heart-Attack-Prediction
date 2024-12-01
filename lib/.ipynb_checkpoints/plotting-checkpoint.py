
def show_metrics(y_test, prediction):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Calcul des métriques
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, zero_division=0)
    recall = recall_score(y_test, prediction, zero_division=0)
    f1 = f1_score(y_test, prediction, zero_division=0)
    # Stocker les métriques dans un dictionnaire
    metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
    }
    # Plot avec Seaborn (optionnel)                            ,
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()),
                hue = list(metrics.keys()),  
                palette="viridis", legend=True)
    plt.ylim(0, 1)
    plt.title("Performance Metrics")
    plt.ylabel("Scores")
    plt.xlabel("Metrics")
    plt.show()
    
    return metrics


def show_crossvalidation_box(model, x_train_scaled, y_train):
    from sklearn.model_selection import cross_val_score, KFold
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    kfold = KFold(n_splits = 7, shuffle=True, random_state = 42)
    scores = cross_val_score(model, x_train_scaled, y_train, cv = kfold)

    #display_scores(scores, 'random forest')
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=scores, color="orange", width=0.4)
    sns.stripplot(data=scores, color="black", alpha=0.7, size=8, jitter=True)
    plt.axhline(y=np.mean(scores), color="red", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
    plt.title("Cross-Validation Scores Distribution")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    return scores
    

def display_scores(scores, name):
    print("--------------------", name, "--------------------")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def show_confusion_matrix(y_test, prediction):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    confusion = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='plasma')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()