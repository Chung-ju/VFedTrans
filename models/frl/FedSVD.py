import numpy as np
import time

def svd(X: np.ndarray):
    m, n = X.shape
    if m >= n:
        return np.linalg.svd(X)
    else:
        u, s, v = np.linalg.svd(X.T)
        return v.T, s, u.T

class FedSVD:
    def __init__(self):
        self.seed = np.random.RandomState()

    def learning(self, Xs: list, block_size=100):
        num_participants = len(Xs)
        
        ground_truth = np.concatenate(Xs, axis=1)
        m, n = ground_truth.shape

        # start_time = time.time()
        P = self.efficient_orthogonal(n=Xs[0].shape[0], block_size=block_size)
        # inter_time = time.time()
        Q = self.efficient_orthogonal(n=np.sum([e.shape[1] for e in Xs]), block_size=block_size)
        # end_time = time.time()
        # print('Time for P: {:.8f}s'.format(inter_time - start_time))
        # print('Time for Q: {:.8f}s'.format(end_time - inter_time))
        cum_lens = np.cumsum([X.shape[1] for X in Xs])
        Qs = np.split(Q, cum_lens[:-1]) 

        X_mask_partitions = []
        for i in range(num_participants):
            X_mask_partitions.append(P @ Xs[i] @ Qs[i])
        X_mask = self.secure_aggregation(X_mask_partitions)

        U_mask, sigma, VT_mask = svd(X_mask)

        U_mask = U_mask[:, :min(m, n)]
        VT_mask = VT_mask[:min(m, n), :]

        U = P.T @ U_mask
        U = np.array(U)
        
        self.Xs_fed = np.matmul(P.transpose(), U)

    def secure_aggregation(self, Xs):
        n = len(Xs)
        size = Xs[0].shape
        perturbations = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(self.random(size))
            perturbations.append(row)
        perturbations = np.array(perturbations)
        perturbations -= np.transpose(perturbations, [1, 0, 2, 3])
        ys = [Xs[i] + np.sum(perturbations[i], axis=0) for i in range(n)]
        return np.sum(ys, axis=0)

    def random(self, size):
        return np.random.randint(low=-10 ** 5, high=10 ** 5, size=size) + np.random.random(size)

    def efficient_orthogonal(self, n, block_size=None):
        if block_size != None:
            qs = [block_size] * (n // block_size)
            if n % block_size != 0:
                qs.append(n % block_size)
            q = np.zeros([n, n])
            start_idx = 0
            for sub_n in qs:
                sub_matrix, _ = np.linalg.qr(np.random.randn(sub_n, sub_n), mode='full')
                q[start_idx:start_idx + sub_n, start_idx:start_idx + sub_n] = sub_matrix
                start_idx += sub_n
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        return q
    
    #  def efficient_orthogonal(self, n, block_size=None):
    #     if block_size != None:
    #         qs = [block_size] * int(n / block_size)
    #         if n % block_size != 0:
    #             qs[-1] += (n - np.sum(qs))
    #         q = np.zeros([n, n])
    #         for i in range(len(qs)):
    #             sub_n = qs[i]
    #             sub_matrix = self.efficient_orthogonal(sub_n, block_size=sub_n)
    #             idx = int(np.sum(qs[:i]))
    #             q[idx:idx + sub_n, idx:idx + sub_n] += sub_matrix
    #     else:
    #         q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
    #     return q

    def get_fed_representation(self):
        return np.array(self.Xs_fed)