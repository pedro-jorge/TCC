from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import sklearn.neural_network
import pandas as pd
from typing import *
from pymoo.core.callback import Callback

def generate_train_val_plot(clf: sklearn.neural_network):
    val_loss = clf.validation_scores_
    train_loss = clf.loss_curve_
    
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss, label='Validation')
    plt.plot(train_loss, label='Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

def train_with_cross_val(clf: sklearn.neural_network, X: pd.DataFrame, y: pd.Series, metrics: List[str], cv: int=5):
    scores = {}
    for metric in metrics:
        scores[metric] = cross_val_score(clf, X, y, cv=cv, scoring=metric)

    return scores


class PymooCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["best_individuals"] = []
    
    def notify(self, algorithm):
        self.data["best_individuals"].append(algorithm.pop.get("F").min())