from math import dist
import numpy as np
import torch

class VFedPCA:
    def __init__(self, **kwargs) -> None:
        self.fed_dis = []   # [[dis_1, dis_2, dis_3, ..., dis_c], ...], c=communication period

    def fed_representation_learning(self, params, data_full, clients_data_list):
        '''
            Step 1: Get Global Maximum Eigenvector By Using the Non-split Dataset
        '''
        data_full = data_full[:, :int(data_full.shape[-1] / params['party_num']) * params['party_num']] # ignore mismatch
        global_eigs, global_eigv = self.local_power_iteration(params, data_full, iter_num=params['iter_num'], com_time=0, warm_start=None) 
        
        '''
            Step 2: Multi-shot Federated Learning with VFedAKPCA
        '''
        torch.cuda.empty_cache
        # for p_idx in range(len(args.p_list)):
        d_list = clients_data_list # [d1, d2, ...d_p], d_p=[n, fea_num]
        p_num = params['party_num']
        ep_list, vp_list = [], []
        print('Before:')
        print('Task party: ', d_list[0].shape)
        print('Data party: ', d_list[1].shape)
        # start multi-shot federated learning
        if params['warm_start']:
            print("Warning: you are using Local Power Iteration with Warm Start!")
        fed_u = None
        for cp in range(params['period_num'] + 1):
            # get the eigenvalue and eigenvector for each client d
            for i in range(p_num):
                ep, vp = self.local_power_iteration(params, d_list[i], iter_num=params['iter_num'], com_time=cp, warm_start=fed_u)
                ep_list.append(ep)
                vp_list.append(vp)
            if cp == 0:
                print("Warning: isolate period!")
                isolate_u = self.isolate(ep_list, vp_list)
                dis_p = self.squared_dis(global_eigv, isolate_u)
                self.fed_dis.append(dis_p)
                continue

            # federated vector
            fed_u = self.federated(ep_list, vp_list, params['weight_scale']) # weight scale method

            # the global vector (from non-split dataset) and federated vector distance
            dis_p = self.squared_dis(global_eigv, fed_u)
            self.fed_dis.append(dis_p)
            
            # reconstruct global
            rs_fed_u = np.expand_dims(fed_u, axis=-1)   # 4000 x 1
            # print('rs_fed_u: ', rs_fed_u.shape)
            # print('data_full: ', data_full.shape)
            mid_up = data_full.T.dot(rs_fed_u)          # X_s x 1
            up_item = mid_up.dot(mid_up.T)              # X_s x X_s
            up_item_norm = up_item / (np.linalg.norm(up_item) + 1e-9)
            data_full = data_full.dot(up_item_norm)     # 4000 x X_s
            # print('data_full: ', data_full.shape)

            # reconstruct local
            for i in range(p_num):
                rs_fed_u = np.expand_dims(fed_u, axis=-1)   # 4000 x 1
                mid_up = d_list[i].T.dot(rs_fed_u)          # X_i x 1
                up_item = mid_up.dot(mid_up.T)              # X_i x X_i
                up_item_norm = up_item / (np.linalg.norm(up_item) + 1e-9)
                d_list[i] = d_list[i].dot(up_item_norm)     # 4000 x X_i
        print('After:')
        print('Task party: ', d_list[0].shape)
        print('Data party: ', d_list[1].shape)

        return data_full

    # Calculate the distance error between 
    # vfed and global eigenvectors (square)
    def squared_dis(self, a, b, r=2.0):
        d = sum(((a[i] - b[i]) ** r) for i in range(len(a))) ** (1.0 / r)

        return d
    
    # Federated algorithm
    def federated(self, ep_list, vp_list, weight_scale):
        # the weight of each client based on eigenvalue
        v_w = ep_list / np.sum(ep_list)
        if weight_scale:
            print("Warning: you are using weight scaling method!")
            eta = np.mean(v_w) #
            en_num = len(ep_list) // 2 # the number of enhance clients
            idx = np.argsort(-v_w) # descending sort
            print("Before: ", v_w) 
            for i in idx[:en_num]:
                v_w[i] = (1 + eta) * v_w[i]
            for j in idx[en_num:]:
                v_w[j] = (1 - eta) * v_w[j]
            print("After: ", v_w)

        # re-weight the importance of each client's eigenvector (v_w * v_arr)
        B = [np.dot(k, v) for k, v in zip(v_w, vp_list)]
        # federated vector u as shared projection feature vector
        u = np.sum(B, axis=0)

        return u

    # isolate algorithm
    def isolate(self, ep_list, vp_list):
        ep_avg = [1.0 for i in range(len(ep_list))]
        # the weight of each client based on eigenvalue
        v_w = ep_avg / np.sum(ep_avg)
        B = [np.dot(k, v) for k, v in zip(v_w, vp_list)]
        # federated vector u as shared projection feature vector
        u = np.sum(B, axis=0)

        return u
    
    # Local power iteration processing or with warm start v 
    def local_power_iteration(self, params, X, iter_num, com_time, warm_start):
        A = np.cov(X)
        # start with a random vector or warm start v
        judge_use = com_time not in [0, 1] and params['warm_start']
        b_k = warm_start if judge_use else np.random.rand(A.shape[0])

        for _ in range(iter_num):
            # eigenvector
            a_bk = np.dot(A, b_k) 
            b_k = a_bk / (np.linalg.norm(a_bk) + 1e-9) #
            
            # eigenvalue
            e_k = np.dot(A, b_k.T).dot(b_k) / np.dot(b_k.T, b_k)

        return e_k, b_k

if __name__ == '__main__':
    X_task = torch.normal(0, 1, size=(500, 1000)).numpy()
    X_data = torch.normal(0, 1, size=(500, 1000)).numpy()
    X = torch.normal(0, 1, size=(500, 1900)).numpy()
    params = {
        'iter_num': 100,
        'party_num': 2,
        'warm_start': False,
        'period_num': 10,
        'weight_scale': False
    }
    model = VFedPCA()
    print(model.fed_representation_learning(params, X, [X_task, X_data]))