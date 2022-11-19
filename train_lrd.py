import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

class LocalRepresentationDistillation():
    def __init__(self, lrd_model, params, device) -> None:
        self.model = lrd_model
        self.params = params
        self.device = device
        self.total_representation = None

    def training_step(self, X_task, X_fed):
        self.X_task = torch.from_numpy(X_task).to(self.device).to(torch.float32)
        X_fed = torch.from_numpy(X_fed).to(self.device)
        if self.model.name == 'AE':
            self.model = self.model.to(self.device)
            self.train_ae(self.X_task, X_fed)
        elif self.model.name == 'BetaVAE':
            self.model = self.model.to(self.device)
            self.train_bvae(self.X_task, X_fed)
        elif self.model.name == 'GAN':
            self.train_gan(self.X_task, X_fed)
    
    def train_ae(self, X_task, X_fed):
        divide_line = X_task.shape[0] - X_fed.shape[0]
        data_holder = torch.utils.data.DataLoader(
            dataset=X_task,
            batch_size=self.params['batch_size'],
            shuffle=False
        )

        mse_loss = nn.MSELoss(reduction='mean')
        mae_loss = nn.L1Loss(reduction='sum')
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])

        for epoch in range(self.params['num_epochs']):
            num_samples = 0
            for _, X in enumerate(data_holder):
                num_samples += X.shape[0]
                encoding, decoding = self.model(X)

                loss = mse_loss(X, decoding) / X.shape[0]

                if self.params['enable_distill_penalty'] and num_samples > divide_line:
                    soft_label = mae_loss(encoding, X_fed[num_samples - X.shape[0] - divide_line:num_samples - divide_line, :]) / X.shape[0]
                    loss += soft_label * self.params['lmd']
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train_bvae(self, X_train, X_fed):
        divide_line = X_train.shape[0] - X_fed.shape[0]
        self.data_holder = torch.utils.data.DataLoader(
            dataset=X_train,
            batch_size=self.params['batch_size'],
            shuffle=False
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'])

        for _ in range(self.params['num_epochs']):
            total_idx = 0
            for _, X in enumerate(self.data_holder):
                total_idx += X.shape[0]
                X = X.to(self.device).view(X.size()[0], -1)
                X = Variable(X).to(torch.float32)
                
                outputs = self.model(X)

                train_loss = self.model.loss_function(*outputs)

                if self.params['enable_distill_penalty'] == True and total_idx > divide_line:
                    # print(self.model.get_hidden_code().shape)
                    # print(X_fed[total_idx - X.shape[0] - divide_line:total_idx - divide_line, :].shape)
                    # print(total_idx - X.shape[0] - divide_line)
                    # print(total_idx, X.shape[0], divide_line)
                    soft_label = torch.abs(self.model.get_hidden_code() - X_fed[total_idx - X.shape[0] - divide_line:total_idx - divide_line, :]).sum()
                    train_loss += soft_label * self.params['lmd']
                
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

    def train_gan(self, X_train, X_fed):
        self.model.train(
            X_train,
            X_fed[:, X_fed.shape[1] - X_train.shape[1]:],
            self.params)


    def representation_distillation_step(self, X_new=None, is_new=False, is_store=False):
        if self.model.name == 'AE':
            representation = self.model.generate(self.X_task) if not is_new else self.model.generate(X_new)
        elif self.model.name == 'BetaVAE':
            representation = self.model.generate(self.X_task) if not is_new else self.model.generate(X_new)
        elif self.model.name == 'GAN':
            representation = self.model.generate_representation('models/lrd/discriminator.pkl', 'models/lrd/generator.pkl')
        
        if not is_new:
            X_rep = np.concatenate([self.X_task.detach().cpu().numpy(), representation], axis=1)
        else:
            X_rep = np.concatenate([X_new.detach().cpu().numpy(), representation], axis=1)

        if is_store:
            self.representation = representation

        return X_rep
    
    def get_representation(self, X_new):
        return np.concatenate([X_new, self.representation], axis=1)