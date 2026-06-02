import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

class Helper():
    def __init__(self):
        # Data directory
        self.DATA_DIR = "./Data"
        if not os.path.isdir(self.DATA_DIR):
            self.DATA_DIR = "../resource/asnlib/publicdata/bankruptcy_data"
        
        self.dataset = "bankruptcy_data.csv"

    def getData(self):
        return np.genfromtxt(os.path.join(self.DATA_DIR, self.dataset), delimiter=',', skip_header=1)
    
    def preprocess_data(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    
    def modelPath(self, modelName):
        return os.path.join(".", "models", modelName)

    def saveModel(self, model, modelName): 
        model_path = self.modelPath(modelName)
        try:
            os.makedirs(model_path)
        except OSError:
            print(f"Directory {model_path} already exists, files will be overwritten.")
        with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved in {model_path}")
    
    def loadModel(self, modelName):
        model_path = self.modelPath(modelName)
        with open(os.path.join(model_path, 'model.pkl'), 'rb') as f:
            return pickle.load(f)
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        if y_prob is not None:
            print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    
    @staticmethod
    def load_data_from_folders(train_folder, holdout_folder):
        train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith(".csv")]
        holdout_files = [os.path.join(holdout_folder, f) for f in os.listdir(holdout_folder) if f.endswith(".csv")]
        
        train_data = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
        holdout_data = pd.concat([pd.read_csv(f) for f in holdout_files], ignore_index=True)
        
        return train_data, holdout_data
