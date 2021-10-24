import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve


def emojiEvaluationGroupedBarChart(df, head=5):
    fig = go.Figure()
    df = df.head(head)
    x_labels = [f'{x}<br>{y}' for x, y in zip(df['model'], df['vectorizer'])]

    fig.add_trace(go.Bar(x=x_labels, y=df['train_acc']*100, name='Train Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=df['test_acc']*100, name='Test Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=df['train_roc_auc']*100, name='Train ROC-AUC'))
    fig.add_trace(go.Bar(x=x_labels, y=df['test_roc_auc']*100, name='Test ROC-AUC'))

    fig.show()
    
def confusionMatrix(y_true, y_pred, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()
    
def rocAuc(model, X_vectorizer, y_true):
    plot_roc_curve(model, X_vectorizer, y_true)
    plt.show()
    
def saveByPickle(object, path):
    pickle.dump(object, open(path, "wb"))
    print(f"{object} has been saved at {path}.")