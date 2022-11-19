import numpy as np
import pandas as pd
import json
import yaml
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from xgboost import XGBClassifier

def load_data(dataset: str):
    if dataset == 'HAPT':
        X_train = np.loadtxt('../../autodl-nas/HAPT/X_train.txt')
        y_train = np.loadtxt('../../autodl-nas/HAPT/y_train.txt')
        X_test = np.loadtxt('../../autodl-nas/HAPT/X_test.txt')
        y_test = np.loadtxt('../../autodl-nas/HAPT/y_test.txt')
        X = np.concatenate([X_train, X_test], axis=0).astype(float)
        y = np.concatenate([y_train, y_test], axis=0).astype(float).reshape([X.shape[0], 1])
    elif dataset == 'TCGA':
        X = np.loadtxt('../../autodl-nas/RNA-Seq/data.csv', delimiter=',')
        y = np.loadtxt('../../autodl-nas/RNA-Seq/labels.csv')
        y = y.reshape((y.shape[0], 1))
    elif dataset == 'MIMIC':
        data = pd.read_csv('../../autodl-nas/MIMIC/mimic3d.csv')
        drop_cols = [
            'LOSgroupNum', 'hadm_id', 'AdmitDiagnosis',
            'AdmitProcedure', 'religion', 'insurance',
            'ethnicity', 'marital_status', 'ExpiredHospital',
            'LOSdays', 'gender', 'admit_type', 'admit_location']
        X = data.drop(drop_cols, axis=1)
        y = data['LOSgroupNum'].to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = y.reshape((len(y), 1))
    elif dataset == 'Breast':
        data = pd.read_csv('dataset/breast/data.csv')
        data.drop(["Unnamed: 32","id"], inplace=True, axis=1)
        data = data.rename(columns={"diagnosis":"target"})
        data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]
        X = data.drop(["target"], axis = 1).to_numpy()
        y = data.target.to_numpy().reshape((len(X), 1))
    else:
        print('Wrong dataset name!')
        return

    return X, y


def load_dataset_config(dataset: str, type: str):
    with open('configs/configuration.json') as load_config:
        config = json.load(load_config)

    return config[dataset][type]

def load_task_config():
    filename = './configs/task_config.yaml'
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def load_model_config(model_name: str):
    filename = './configs/' + model_name.lower() + '.yaml'
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_gpu(gpu_idx='0'):
    return torch.device('cuda:' + gpu_idx if torch.cuda.is_available() else "cpu")

def svd(X: np.ndarray):
    m, n = X.shape
    if m >= n:
        return np.linalg.svd(X)
    else:
        u, s, v = np.linalg.svd(X.T)
        return v.T, s, u.T

def equal_split(dataset='MIMIC', start_sample=0, start_feature=0, shared_sample=None, shared_feature=None, num_sample=None, num_feature=None):
    X, y = load_data(dataset)
    setting = load_dataset_config(dataset, 'equal_split')
    
    num_sample = setting['num_sample'] if num_sample == None else num_sample
    num_feature = setting['num_feature'] if num_feature == None else num_feature
    shared_sample = setting['shared_sample'] if shared_sample == None else shared_sample
    shared_feature = setting['shared_feature'] if shared_feature == None else shared_feature

    # Task party
    X_task = X[start_sample:start_sample + num_sample, start_feature:start_feature + num_feature]
    y_task = y[start_sample:start_sample + num_sample, :]

    # Data party
    X_data = X[start_sample + num_sample - shared_sample:start_sample + 2 * num_sample - shared_sample, start_feature + num_feature - shared_feature:start_feature + 2 * num_feature - shared_feature]
    y_data = y[start_sample + num_sample - shared_sample:start_sample + 2 * num_sample - shared_sample, :]

    # Shared samples
    task_shared = X_task[num_sample - shared_sample:num_sample, :]
    data_shared = X_data[:shared_sample, shared_feature:num_feature]
    X_shared = np.concatenate([task_shared, data_shared], axis=1)
    
    X = X[start_sample:start_sample + num_sample, :]
    y = y[start_sample:start_sample + num_sample, :]

    return X_task, y_task, X_shared, X_data

def unequal_split(dataset='MIMIC', start=0, task_feature=None, data_feature=None, shared_sample=None, shared_feature=None):
    X, y = load_data(dataset)
    setting = load_dataset_config(dataset, 'unequal_split')

    task_num_sample = setting['task_num_sample']
    data_num_sample = setting['data_num_sample']
    task_num_feature = setting['task_num_feature'] if task_feature == None else task_feature
    data_num_feature = setting['data_num_feature'] if data_feature == None else data_feature
    shared_sample = setting['shared_sample'] if shared_sample == None else shared_sample
    shared_feature = setting['shared_feature'] if shared_feature == None else shared_feature

    # Task party
    X_task = X[start:start + task_num_sample, :task_num_feature]
    y_task = y[start:start + task_num_sample, :]

    # Data party
    X_data = X[start + task_num_sample - shared_sample:start + task_num_sample + data_num_sample - shared_sample,
         task_num_feature - shared_feature:task_num_feature + data_num_feature - shared_feature]
    y_data = y[start + task_num_sample - shared_sample:start + task_num_sample + data_num_sample - shared_sample, :]

    # Shared samples
    task_shared = X_task[task_num_sample - shared_sample:task_num_sample, :]
    data_shared = X_data[:shared_sample, shared_feature:data_num_feature]
    shared = np.concatenate([task_shared, data_shared], axis=1)

    X = X[start:start + task_num_sample, :]
    y = y[start:start + task_num_sample, :]

    return X_task, y_task, shared, X_data, y_data, X, y

def imbalanced_split(dataset='MIMIC', type='iid', select_num=1, noniid_size=100):
    X, y = load_data(dataset)
    data = pd.DataFrame(np.concatenate([X, y], axis=1))
    setting = load_dataset_config(dataset, 'imbalanced_split')
    task_num_sample = setting['task_num_sample']
    data_num_sample = setting['data_num_sample']
    task_num_feature = setting['task_num_feature']
    data_num_feature = setting['data_num_feature']
    shared_sample = setting['shared_sample']
    shared_feature = setting['shared_feature']
    if type == 'iid':
        pass
    else:
        for i in range(select_num):
            select_label = data[data.shape[1] - 1].max()
            if i == 0:
                data_new = data[data[data.shape[1] - 1] == select_label].to_numpy()
            else:
                data_new = np.concatenate([data_new, data[data[data.shape[1] - 1] == select_label].to_numpy()], axis=0)
            data = data[data[data.shape[1] - 1] != select_label]
        data = data.to_numpy()
        print(data_new.shape)
        np.random.shuffle(data_new)
        data = np.concatenate([data, data_new[noniid_size:, :]], axis=0)
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1:].astype(int)
        print(X.shape)

        # Task party
        X_task = X[:task_num_sample, :task_num_feature]
        y_task = y[:task_num_sample, :]

        # Data party
        X_data = X[task_num_sample - shared_sample:task_num_sample + data_num_sample - shared_sample,
            task_num_feature - shared_feature:task_num_feature + data_num_feature - shared_feature]

        # Shared samples
        task_shared = X_task[task_num_sample - shared_sample:task_num_sample, :]
        data_shared = X_data[:shared_sample, shared_feature:data_num_feature]
        shared = np.concatenate([task_shared, data_shared], axis=1)

        # New samples
        data_new = data_new[:noniid_size, :]
        X_new, y_new = data_new[:, :task_num_feature], data_new[:, -1:].astype(int)
    
    print(X_task.shape)
    print(y_task.shape)
    print(shared.shape)
    print(X_new.shape)
    print(y_new.shape)

    return X_task, y_task, shared, X_data, X_new, y_new
    

def method_select(method_name):
    if method_name == 'Logistic':
        model = LogisticRegression()
    elif method_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=8)
    elif method_name == 'XGBoost':
        model = XGBClassifier()
    elif method_name == 'SVM':
        model = SVC(gamma='scale')
    elif method_name == 'Neural Network':
        model = MLPClassifier(hidden_layer_sizes=(100, 100, 50), alpha=0.01, max_iter=400)
    elif method_name == 'AdaBoost':
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=100, algorithm="SAMME.R", learning_rate=0.5)
    elif method_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=200, max_depth=20)
    else:
        print('Wrong method name!')

    return model

if __name__ == '__main__':
    arr = np.arange(9).reshape((3, 3))
    np.random.shuffle(arr)
    print(arr)
