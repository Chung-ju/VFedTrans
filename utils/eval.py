import numpy as np
import torch
import torch.utils
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.append('..')
from models import *


class ModelSearch:
    def __init__(self):
        self.models_params = {
            "knn": {
                "model": KNeighborsClassifier(),
                "param_grid": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
            },
            "xgboost": {
                "model": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    # "subsample": [0.6, 0.8, 1.0]
                },
            },
            "svc": {
                "model": SVC(probability=True, random_state=42),
                "param_grid": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"]
                },
            },
            "mlp": {
                "model": MLPClassifier(random_state=42, max_iter=500),
                "param_grid": {
                    # "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    # "activation": ["relu", "tanh"],
                    # "solver": ["adam", "sgd"],
                    "alpha": [0.0001, 0.001, 0.01]
                },
            },
            "rf": {
                "model": RandomForestClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "adaboost": {
                "model": AdaBoostClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                },
            },
        }
    
    def get_model_and_params(self, model_name):
        return self.models_params.get(model_name, None)
    
    def search(self, model_name, X, y, search_type="grid", cv=5, n_iter=50, scoring="accuracy", random_state=42):
        model_info = self.get_model_and_params(model_name)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = model_info["model"]
        param_grid = model_info["param_grid"]

        if search_type == "grid":
            searcher = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, verbose=2, n_jobs=-1)
        elif search_type == "random":
            searcher = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=cv, scoring=scoring, verbose=2, random_state=random_state, n_jobs=-1)
        else:
            raise ValueError(f"Search type '{search_type}' is not supported. Use 'grid' or 'random'.")

        searcher.fit(X, y)
        return searcher.best_params_, searcher.best_score_

def select_model(model_name):
    if model_name == 'knn':
        model = KNeighborsClassifier()
    elif model_name == 'xgboost':
        model = XGBClassifier()
    elif model_name == 'svc':
        model = SVC(gamma='scale')
    elif model_name == 'mlp':
        model = MLPClassifier()
    elif model_name == 'adaboost':
        model = AdaBoostClassifier()
    elif model_name == 'rf':
        model = RandomForestClassifier()
    else:
        print('Wrong method name!')

    return model
    
def select_tabnet_model(X_train, y_train, X_test, device):
    num_classes = len(np.unique(y_train))
    num_epochs = 50
    batch_size = 128
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).squeeze().to(device)
    # y_train_tensor = torch.nn.functional.one_hot(torch.tensor(y_train, dtype=torch.int64).squeeze(), num_classes=num_classes).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    print(num_classes)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        X_test_tensor,
        batch_size=batch_size,
        shuffle=False
    )
    
    # print(f"X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")
    
    model = TabNet(input_dim=X_train_tensor.shape[1], output_dim=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
            for X_batch, y_batch in t:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.long())
                loss.backward()
                optimizer.step()
                
                t.set_postfix(loss=loss.item())
    
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch in test_loader:
            y_pred = model(X_batch)
            y_pred_list.append(torch.argmax(y_pred, dim=1).cpu().detach().numpy())
    
    return np.concatenate(y_pred_list, axis=0)

def select_fcnn_model(X_train, y_train, X_test, device):
    num_classes = len(np.unique(y_train))
    num_epochs = 50
    batch_size = 128
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).squeeze().to(device)
    # y_train_tensor = torch.nn.functional.one_hot(torch.tensor(y_train, dtype=torch.int64).squeeze(), num_classes=num_classes).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        X_test_tensor,
        batch_size=batch_size,
        shuffle=False
    )
    
    model = FCNN(input_dim=X_train_tensor.shape[1], output_dim=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
            for X_batch, y_batch in t:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.long())
                loss.backward()
                optimizer.step()
                
                t.set_postfix(loss=loss.item())
    
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch in test_loader:
            y_pred = model(X_batch)
            y_pred_list.append(torch.argmax(y_pred, dim=1).cpu().detach().numpy())
    
    return np.concatenate(y_pred_list, axis=0)

def select_wideanddeep_model(X_train, y_train, X_test, device):
    num_classes = len(np.unique(y_train))
    num_epochs = 50
    batch_size = 128
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).squeeze().to(device)
    # y_train_tensor = torch.nn.functional.one_hot(torch.tensor(y_train, dtype=torch.int64).squeeze(), num_classes=num_classes).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        X_test_tensor,
        batch_size=batch_size,
        shuffle=False
    )
    
    model = WideAndDeep(input_dim=X_train_tensor.shape[1], output_dim=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
            for X_batch, y_batch in t:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.long())
                loss.backward()
                optimizer.step()
                
                t.set_postfix(loss=loss.item())
    
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch in test_loader:
            y_pred = model(X_batch)
            y_pred_list.append(torch.argmax(y_pred, dim=1).cpu().detach().numpy())
    
    return np.concatenate(y_pred_list, axis=0)

def select_cnn_model(X_train, X_aug_train, y_train, X_test, X_aug_test, y_test, device):
    batch_size = 64
    num_epochs = 50
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_aug_train_tensor = torch.tensor(X_aug_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).squeeze().to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_aug_test_tensor = torch.tensor(X_aug_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64).squeeze().to(device)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, X_aug_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, X_aug_test_tensor, y_test_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    # model = CNN(X_train.shape[2], 0).to(device)
    model = CNN(X_train.shape[2], X_aug_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    for _ in range(num_epochs):
        with tqdm(train_loader, unit='batch') as tepoch:
            for data in tepoch:
                inputs, inputs_aug, labels = data
                inputs, inputs_aug, labels = inputs.to(device), inputs_aug.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, inputs_aug)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            inputs, inputs_aug, labels = data
            inputs, inputs_aug, labels = inputs.to(device), inputs_aug.to(device), labels.to(device)
            outputs = model(inputs, inputs_aug)
            predicted = torch.round(outputs.squeeze())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))


def select_vgg16_model(X_train, X_aug_train, y_train, X_test, X_aug_test, y_test, device):
    batch_size = 64
    num_epochs = 100
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_aug_train_tensor = torch.tensor(X_aug_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).squeeze().to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_aug_test_tensor = torch.tensor(X_aug_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64).squeeze().to(device)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, X_aug_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, X_aug_test_tensor, y_test_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    
    # model = CNN(X_train.shape[2], 0).to(device)
    model = VGG16(X_train.shape[2], X_aug_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    for _ in range(num_epochs):
        with tqdm(train_loader, unit='batch') as tepoch:
            for data in tepoch:
                inputs, inputs_aug, labels = data
                inputs, inputs_aug, labels = inputs.to(device), inputs_aug.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, inputs_aug)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            inputs, inputs_aug, labels = data
            inputs, inputs_aug, labels = inputs.to(device), inputs_aug.to(device), labels.to(device)
            outputs = model(inputs, inputs_aug)
            predicted = torch.round(outputs.squeeze())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))

class DownstreamTask():
    def __init__(self, train_ratio=0.8, random_state=101) -> None:
        self.train_ratio = train_ratio
        self.random_state = random_state

    def oversample(self, X, y):
        smote = SMOTE(random_state=self.random_state)
        X, y = smote.fit_resample(X, y)
        return X, y
    
    def run_img(self, X, X_aug, y, model, device):
        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(indices, train_size=self.train_ratio, random_state=self.random_state)
        
        X_train, X_test = X[train_indices], X[test_indices]
        X_aug_train, X_aug_test = X_aug[train_indices], X_aug[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        if model == 'cnn':
            select_cnn_model(X_train, X_aug_train, y_train, X_test, X_aug_test, y_test, device)
        elif model == 'vgg16':
            select_vgg16_model(X_train, X_aug_train, y_train, X_test, X_aug_test, y_test, device)
        
    def run(self, X, y, dataset, model_name, device, search=True):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.ravel(), train_size=self.train_ratio, random_state=self.random_state
        )
        
        if model_name == 'tabnet':
            y_pred = select_tabnet_model(X_train, y_train, X_test, device)
        elif model_name == 'fcnn':
            y_pred = select_fcnn_model(X_train, y_train, X_test, device)
        elif model_name == 'wideanddeep':
            y_pred = select_wideanddeep_model(X_train, y_train, X_test, device)
        else:
            if search:
                model_search = ModelSearch()
                best_params, _ = model_search.search(model_name, X_train, y_train, search_type="grid", cv=5)
                print(f"Best parameters: {best_params}")
                
                model_info = model_search.get_model_and_params(model_name)
                model = model_info["model"].set_params(**best_params)
            else:
                model = select_model(model_name)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        results = self.evaluate_model(y_pred, y_test)
        print(f"Test evaluation metrics: {results}")
    
    def evaluate_model(self, y_pred, y_test):
        """
        Evaluate the model using test set.

        Params:
        - y_pred: predicted labels
        - y_test: test set labels

        Returns:
        - results: dict, including evaluation metrics
        """
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }
        return results       
        