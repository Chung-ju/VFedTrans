from copy import deepcopy
from typing import List, Tuple
from uuid import uuid4
import yaml
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import os
import random
import numpy as np
import pandas as pd
import torch

def load_partition_params(dataset: str):
    with open('configs/partition.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config[dataset]


def load_dataset(dataset='mimic') -> Tuple[np.ndarray, np.ndarray]:
    dataset_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'dataset')
    if dataset == 'mimic':
        data = pd.read_csv(os.path.join(dataset_dir, 'mimic/mimic3d.csv'))
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
    elif dataset == 'rna':
        X = pd.read_csv(os.path.join(dataset_dir, 'rna/data.csv')).values[:, 1:].astype(np.float32)
        y = pd.read_csv(os.path.join(dataset_dir, 'rna/labels.csv')).values[:, 1]
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = y.reshape((y.shape[0], 1))
    elif dataset == 'cardio':
        data = pd.read_csv("dataset/cardio/cardio_train.csv", sep=";")
        data.drop("id", axis=1, inplace=True)
        data.drop_duplicates(inplace=True)
        data["bmi"] = data["weight"] / (data["height"] / 100) ** 2
        out_filter = ((data["ap_hi"] > 250) | (data["ap_lo"] > 200))
        data = data[~out_filter]
        out_filter2 = ((data["ap_hi"] < 0) | (data["ap_lo"] < 0))
        data = data[~out_filter2]
        y = data["cardio"].values
        X = data.drop("cardio", axis=1).values
    elif dataset == 'heart':
        data = pd.read_csv('dataset/heart/heart_attack_prediction_dataset.csv')
        data['BP_Systolic'] = data['Blood Pressure'].apply(lambda x: x.split('/')[0])
        data['BP_Diastolic'] = data['Blood Pressure'].apply(lambda x: x.split('/')[1])
        ordinal_map = {'Healthy':2,'Average':1,'Unhealthy':0}
        data['Diet'] = data['Diet'].map(ordinal_map)
        data = pd.get_dummies(data, columns = ['Sex'])
        cat_columns = ['Sex_Female','Sex_Male','BP_Systolic','BP_Diastolic']
        data[cat_columns] = data[cat_columns].astype(int)
        X = data[['Age', 'Cholesterol', 'Heart Rate',
                'Diabetes', 'Family History', 'Smoking', 'Obesity',
                'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
                'Previous Heart Problems', 'Medication Use', 'Stress Level',
                'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                'Physical Activity Days Per Week', 'Sleep Hours Per Day',
                'BP_Systolic', 'BP_Diastolic','Sex_Female',
                'Sex_Male']]
        y = data['Heart Attack Risk'].values
        sc = StandardScaler()
        X = sc.fit_transform(X)
    elif dataset == 'sepsis':
        data = pd.read_csv("dataset/sepsis/processed_data.csv")
        X = data.drop('SepsisLabel', axis=1)
        X.drop(['0', '1'], axis=1, inplace=True)
        X = X.values
        y = data['SepsisLabel'].values
    elif dataset == 'leukemia' or dataset == 'pneumonia':
        data = np.load(f'dataset/{dataset}/data.npz')
        X = data['image']
        y = data['label']
    
    return X, y


def partition(dataset: str, partition_params: dict, partition_type='intra', shuffle: bool=False):
    X, y = load_dataset(dataset)
    
    if shuffle:
        sample_indices = np.random.permutation(len(X))
        feature_indices = np.random.permutation(X.shape[1])
        X = X[sample_indices][:, feature_indices]
        y = y[sample_indices]
        
    ids = np.array([str(uuid4()) for _ in range(len(y))])
    original_data = VFedDataset(X=X, y=y, ids=ids)
    
    sample_size = np.arange(len(X))
    feature_size = np.arange(X.shape[1])
    
    print(X.shape)
    print(partition_params)
    
    nl_tp_sample_ids = sample_size[:partition_params['nl_tp_sample']]
    nl_tp_feature_ids = feature_size[:partition_params['nl_tp_feature']]
    
    ol_sample_ids = sample_size[partition_params['nl_tp_sample']:partition_params['nl_tp_sample']+partition_params['ol_sample']]
    ol_tp_feature_ids = feature_size[:partition_params['ol_tp_feature']]
    ol_dp_feature_ids = feature_size[partition_params['ol_tp_feature']:partition_params['ol_tp_feature']+partition_params['ol_dp_feature']]
    # if partition_type == 'intra':
    #     ol_tp_feature_ids = feature_size[:partition_params['ol_tp_feature']]
    #     ol_dp_feature_ids = feature_size[partition_params['ol_tp_feature']:partition_params['ol_tp_feature']+partition_params['ol_dp_feature']]
    # elif partition_type == 'cross':
    #     ol_tp_feature_ids = feature_size[partition_params['nl_tp_feature']:partition_params['nl_tp_feature']+partition_params['ol_tp_feature']]
    #     ol_dp_feature_ids = feature_size[partition_params['nl_tp_feature']+partition_params['ol_tp_feature']:partition_params['nl_tp_feature']+partition_params['ol_tp_feature']+partition_params['ol_dp_feature']]
    
    ol_tp_data = original_data.filter_by_idxs(ol_sample_ids, ol_tp_feature_ids)
    ol_dp_data = original_data.filter_by_idxs(ol_sample_ids, ol_dp_feature_ids)
    nl_tp_data = original_data.filter_by_idxs(nl_tp_sample_ids, nl_tp_feature_ids)
    
    print('Overlapping data of task party:', ol_tp_data.get_data().shape)
    print('Overlapping data of data party:', ol_dp_data.get_data().shape)
    print('Non-overlapping data of task party:', nl_tp_data.get_data().shape)
    
    tp_dict = {
        'ol_sample': ol_tp_data.get_data(),
        'ol_label': ol_tp_data.get_labels(),
        'ol_ids': ol_tp_data.get_ids(),
        'nl_sample': nl_tp_data.get_data(),
        'nl_label': nl_tp_data.get_labels(),
        'nl_ids': nl_tp_data.get_ids()
    }
    
    dp_dict = {
        'ol_sample': ol_dp_data.get_data(),
        'ol_ids': ol_dp_data.get_ids(),
    }

    tp_file_name = 'tp_intra.npy'
    dp_file_name = 'dp_intra.npy'

    np.save(os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', dataset, tp_file_name), tp_dict)
    np.save(os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', dataset, dp_file_name), dp_dict)

def noniid_partition(dataset: str, partition_params: dict, shuffle: bool=True):
    X, y = load_dataset(dataset)
    
    unique_classes = np.unique(y)
    
    if len(unique_classes) == 2:
        indices_0 = np.where(y == 0)[0]
        indices_1 = np.where(y == 1)[0]
        np.random.shuffle(indices_0)
        split_index = len(indices_0) // 2
        indices_0_group1 = indices_0[:split_index]
        indices_0_group2 = indices_0[split_index:]
        indices_group1 = np.concatenate([indices_0_group1, indices_1])
        indices_group2 = indices_0_group2
        X_group1, y_group1 = X[indices_group1], y[indices_group1]
        X_group2, y_group2 = X[indices_group2], y[indices_group2]
    else:
        np.random.shuffle(unique_classes)
        group1_classes = unique_classes[:len(unique_classes) // 2 + 1]
        group2_classes = unique_classes[len(unique_classes) // 2 + 1:]
        
        group1_indices = np.isin(y.ravel(), group1_classes)
        group2_indices = np.isin(y.ravel(), group2_classes)
        
        X_group1, y_group1 = X[group1_indices], y[group1_indices]
        X_group2, y_group2 = X[group2_indices], y[group2_indices]
    
    if shuffle:
        group1_indices_shuffle = np.random.permutation(len(X_group1))
        group2_indices_shuffle = np.random.permutation(len(X_group2))
        X_group1 = X_group1[group1_indices_shuffle]
        y_group1 = y_group1[group1_indices_shuffle]
        X_group2 = X_group2[group2_indices_shuffle]
        y_group2 = y_group2[group2_indices_shuffle]
    
    nl_tp_data = X_group1[:partition_params['nl_tp_sample'], :partition_params['nl_tp_feature']]
    nl_tp_label = y_group1[:partition_params['nl_tp_sample']]
    label_encoder = LabelEncoder()
    nl_tp_label = label_encoder.fit_transform(nl_tp_label)
    
    ol_tp_data = X_group2[:partition_params['ol_sample'], :partition_params['ol_tp_feature']]
    ol_tp_label = y_group2[:partition_params['ol_sample']]
    ol_dp_data = X_group2[:partition_params['ol_sample'], partition_params['ol_tp_feature']:partition_params['ol_tp_feature']+partition_params['ol_dp_feature']]

    tp_dict = {
        'ol_sample': ol_tp_data,
        'ol_label': ol_tp_label,
        'nl_sample': nl_tp_data,
        'nl_label': nl_tp_label,
    }
    
    dp_dict = {
        'ol_sample': ol_dp_data,
    }
    
    tp_file_name = 'tp_cross.npy'
    dp_file_name = 'dp_cross.npy'
    np.save(os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', dataset, tp_file_name), tp_dict)
    np.save(os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', dataset, dp_file_name), dp_dict)
    
def img_partition(dataset: str, partition_params: dict, shuffle: bool=True):
    X, y = load_dataset(dataset)
    if shuffle:
        sample_indices = np.random.permutation(len(X))
        X = X[sample_indices]
        y = y[sample_indices]
        
    print(X.shape)
    print(y.shape)
    
    start_point = X.shape[2] // 2 - partition_params['nl_tp_feature'] // 2
    start_point_1 = X.shape[3] // 2 - partition_params['nl_tp_feature'] // 2
    
    nl_tp_data = X[:partition_params['nl_tp_sample'], :, start_point:start_point+partition_params['nl_tp_feature'], start_point_1:start_point_1+partition_params['nl_tp_feature']]
    nl_tp_label = y[:partition_params['nl_tp_sample']]
    
    ol_data = X[partition_params['nl_tp_sample']:partition_params['nl_tp_sample']+partition_params['ol_sample']]
    ol_tp_data = ol_data[:, :, :, :X.shape[3] // 2]
    ol_dp_data = ol_data[:, :, :, X.shape[3] // 2:]
    ol_tp_label = y[partition_params['nl_tp_sample']:partition_params['nl_tp_sample']+partition_params['ol_sample']]
    
    print('Overlapping data of task party:', ol_tp_data.shape, ol_tp_label.shape)
    print('Overlapping data of data party:', ol_dp_data.shape)
    print('Non-overlapping data of task party:', nl_tp_data.shape, nl_tp_label.shape)
    
    tp_dict = {
        'ol_sample': ol_tp_data.reshape((ol_tp_data.shape[0], -1)),
        'ol_label': ol_tp_label,
        'nl_sample': nl_tp_data,
        'nl_label': nl_tp_label,
    }
    
    dp_dict = {
        'ol_sample': ol_dp_data.reshape((ol_dp_data.shape[0], -1)),
    }
    
    print(tp_dict['ol_sample'].shape)
    print(dp_dict['ol_sample'].shape)
    
    tp_file_name = 'tp_cross.npy'
    dp_file_name = 'dp_cross.npy'
    np.save(os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', dataset, tp_file_name), tp_dict)
    np.save(os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', dataset, dp_file_name), dp_dict)
    
class VFedDataset(Dataset):
    def __init__(self, X=None, y=None, ids=None) -> None:
        super().__init__()
        self.X, self.y, self.ids = X, y, ids
        
    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.float)
        ids = self.ids[index]
        
        if self.y is not None:
            y = torch.tensor(self.y[index])
            return X, y, ids
        else:
            return X, ids
    
    def __len__(self):
        return len(self.X)
    
    def get_data(self):
        return self.X
    
    def get_labels(self):
        if self.y is not None:
            return self.y
        else:
            raise ValueError("No labels available")
    
    def get_ids(self):
        return self.ids
    
    def filter_by_idxs(self, sample_idxs, feature_idxs):
        new_X = self.X[sample_idxs][:, feature_idxs]
        new_y = self.y[sample_idxs]
        new_ids = self.ids[sample_idxs]
        
        return VFedDataset(X=new_X, y=new_y, ids=new_ids)
    