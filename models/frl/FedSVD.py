import numpy as np
import sys
sys.path.append('../..')
from utils import svd

class FedSVD:
    def __init__(self, num_features, num_participants=2, random_seed=100):
        self.num_participants = num_participants
        self.num_features = num_features
        self.seed = np.random.RandomState(random_seed)

    def load_data(self, X: np.ndarray):
        self.X = X
        self.Xs = [self.X[:, e * self.num_features: e * self.num_features + self.num_features] for e in range(self.num_participants)]


    def learning(self):
        ground_truth = np.concatenate(self.Xs, axis=1)
        m, n = ground_truth.shape

        P = self.efficient_orthogonal(n=self.X.shape[0])
        Q = self.efficient_orthogonal(n=np.sum([e.shape[1] for e in self.Xs]))
        Qs = [Q[e * self.num_features: e * self.num_features + self.num_features] for e in range(self.num_participants)]

        X_mask_partitions = []
        for i in range(self.num_participants):
            X_mask_partitions.append(P @ self.Xs[i] @ Qs[i])
        X_mask = self.secure_aggregation(X_mask_partitions)

        U_mask, sigma, VT_mask = svd(X_mask)

        U_mask = U_mask[:, :min(m, n)]
        VT_mask = VT_mask[:min(m, n), :]

        U = P.T @ U_mask

        VTs = []
        k = 1
        transferred_variables = []
        for i in range(self.num_participants):
            Q_i = Qs[i].T
            R1_i = self.seed.random([n, k])
            R2_i = self.seed.random([Q_i.shape[1] + k, Q_i.shape[1] + k])
            Q_i_mask = np.concatenate([Q_i, R1_i], axis=-1) @ R2_i
            VT_i_mask = VT_mask @ Q_i_mask
            VTs.append((VT_i_mask @ np.linalg.inv(R2_i))[:, :Q_i.shape[1]])
            transferred_variables.append([Q_i_mask, VT_i_mask])

        U = np.array(U)
        VTs = np.concatenate(VTs, axis=1)
        # self.Xs_fed = U[:, :min(m, n)] @ np.diag(sigma) @ VTs[:min(m, n), :]
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
            qs = [block_size] * int(n / block_size)
            if n % block_size != 0:
                qs[-1] += (n - np.sum(qs))
            q = np.zeros([n, n])
            for i in range(len(qs)):
                sub_n = qs[i]
                sub_matrix = self.efficient_orthogonal(sub_n, block_size=sub_n)
                idx = int(np.sum(qs[:i]))
                q[idx:idx + sub_n, idx:idx + sub_n] += sub_matrix
        else:
            q, _ = np.linalg.qr(np.random.randn(n, n), mode='full')
        return q


    def get_fed_representation(self):
        return np.array(self.Xs_fed)