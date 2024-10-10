import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(label, anomaly_prediction, plot=False):
    precision = precision_score(label, anomaly_prediction, pos_label=-1)
    recall = recall_score(label, anomaly_prediction, pos_label=-1)
    f1 = f1_score(label, anomaly_prediction, pos_label=-1)
    
    if plot:
        confusion = confusion_matrix(label, anomaly_prediction)
        target_names = ['Anomaly', 'Normal']
        print('\nClassification Report: \n')
        print(classification_report(label, anomaly_prediction, target_names=target_names))

        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
        labels = ['Anomaly', 'Normal']
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=axs[0], xticklabels=labels, yticklabels=labels)
        axs[0].set_xlabel('Predicted Label')
        axs[0].set_ylabel('True Label')
        axs[0].set_title('Confusion Matrix')

        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [precision, recall, f1]
        axs[1].bar(metrics, values)
        axs[1].set_xlabel('Metrics')
        axs[1].set_ylabel('Values')
        axs[1].set_title('Evaluation Metrics')
        axs[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    return precision, recall, f1

def plot_roc_curve(label, anomaly_scores, plot=False):
    fpr, tpr, thresholds = roc_curve(label, anomaly_scores)
    roc_auc = auc(fpr, tpr)

    if plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')   
        plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label='Perfect Classifier')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc="lower right", fontsize=8)
        plt.show()
    return fpr, tpr, roc_auc
    
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Função para plotar t-SNE
def plot_anomalies_tsne(data, anomaly_indexes):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)
    anomaly_data = tsne_data[anomaly_indexes]
    
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c='blue', alpha=0.3, s=2, marker='o', label='Normal')
    plt.scatter(anomaly_data[:, 0], anomaly_data[:, 1], c='red', alpha=0.7, s=2, marker='o', label='Anomaly')
    plt.legend()
    plt.title('t-SNE Anomaly Detection')
    plt.show()

# Função para plotar histograma com threshold
def plot_histogram_with_threshold(scores, threshold):
    plt.figure(figsize=(4, 3))
    plt.hist(scores, bins=80, density=True, alpha=0.6, color='g', label='Anomaly Scores')
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
    plt.xlabel('Anomaly Scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Função para plotar curva ROC
#def plot_roc_curve(fpr, tpr, auc_value):
#    plt.figure(figsize=(4, 4))
#    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_value:.2f})')
#    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')
#    plt.xlim([-0.05, 1.05])
#    plt.ylim([-0.05, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.legend(loc="lower right")
#    plt.show()
