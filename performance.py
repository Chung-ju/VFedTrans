import numpy as np
from utils import method_select
from sklearn.model_selection import train_test_split

class Performance:
    def __init__(self) -> None:
        self.train_size = 0.8
        self.random_state = 101
        self.N = 10

    def run(self, X, y, method_name='Adaboost'):
        self.method = method_select(method_name)
        res = []
        for _ in range(self.N):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size)
            self.method.fit(X_train, y_train)
            res.append(self.method.score(X_test, y_test))
        print(res)
        print(np.mean(res))
    
    def run_new(self, X_new, y_new):
        res = []
        for _ in range(self.N):
            res.append(self.method.score(X_new, y_new))
        print(res)
        print(np.mean(res))
    
    def few_shot(self, X, y, method_name='Adaboost', proportion=0.1):
        self.method = method_select(method_name)
        res = []
        for _ in range(self.N):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size)
            X_train, y_train = X_train[:int(len(X_train) * proportion)], y_train[:int(len(X_train) * proportion)]
            X_test, y_test = X_test[:1000], y_test[:1000]
            # print(X_train.shape)
            # print(X_test.shape)
            self.method.fit(X_train, y_train)
            res.append(self.method.score(X_test, y_test))
        print(res)
        print(np.mean(res))


if __name__ == '__main__':
    X = np.zeros((100, 5))
    y = np.zeros((100, 1))
    p = Performance()
    p.few_shot(X, y)
