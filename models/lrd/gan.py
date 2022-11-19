import os
from random import sample
import torch
import torch.nn as nn
from torchvision import utils
from torch.autograd import Variable


class GAN(object):
    def __init__(self,
                 feature_dim: int,
                 latent_dim: int,
                 d_depth: int=4,
                 g_depth: int=4,
                 negative_slope: float=0.2,
                 **kwargs) -> None:

        # Generator architecture
        self.g_input = feature_dim // 3
        g_output = feature_dim
        increment = (g_output - self.g_input) // (g_depth - 1)
        g_layers = [self.g_input + increment * i for i in range(g_depth - 1)] + [g_output]
        self.G = nn.Sequential()
        
        for i in range(g_depth - 1):
            self.G.add_module('G-linear-' + str(i), nn.Linear(g_layers[i], g_layers[i + 1]))
            self.G.add_module('G-activation-' + str(i), nn.LeakyReLU(negative_slope))
            if i == g_depth - 2:
                self.G.add_module('G-last', nn.Tanh())

        # Discriminator architecture
        d_input = feature_dim
        d_output = 1
        decrement = (d_input - d_output) // (d_depth - 1)
        d_layers = [d_input - decrement * i for i in range(d_depth - 1)] + [d_output]
        self.D = nn.Sequential()
        for i in range(d_depth - 1):
            self.D.add_module('D-linear-' + str(i), nn.Linear(d_layers[i], d_layers[i + 1]))
            self.D.add_module('D-activation-' + str(i), nn.LeakyReLU(negative_slope))
            if i == d_depth - 2:
                self.D.add_module('D-last', nn.Sigmoid())

    def train(self, X_task, X_fed, training_params):
        self.device = X_task.device
        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)
        self.num_samples = X_task.shape[0]

        self.training_params = training_params
        divide_line = X_task.shape[0] - X_fed.shape[0]

        data_holder = torch.utils.data.DataLoader(
            dataset=X_task,
            batch_size=self.training_params['batch_size'],
            shuffle=False
        )
        
        # Binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.training_params['d_LR'], weight_decay=self.training_params['d_WD'])
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.training_params['g_LR'], weight_decay=self.training_params['g_WD'])

        for _ in range(self.training_params['num_epochs']):
            total_idx = 0
            for _, data in enumerate(data_holder):
                total_idx += data.shape[0]
                z = torch.rand((self.training_params['batch_size'], self.g_input))

                real_labels = Variable(torch.ones(self.training_params['batch_size'])).to(self.device)
                fake_labels = Variable(torch.zeros(self.training_params['batch_size'])).to(self.device)
                data, z = Variable(data.to(self.device)), Variable(z.to(self.device))

                # Train discriminator
                # compute BCE_Loss using real data where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # [Training discriminator = Maximizing discriminator being correct]
                outputs = self.D(data)
                d_loss_real = self.loss(outputs.flatten(), real_labels)
                real_score = outputs

                # Compute BCELoss using fake data
                fake_data = self.G(z)
                outputs = self.D(fake_data)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                fake_score = outputs

                # Optimizie discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                z = Variable(torch.randn(self.training_params['batch_size'], self.g_input).to(self.device))
                fake_data = self.G(z)
                outputs = self.D(fake_data)

                g_loss = self.loss(outputs.flatten(), real_labels)

                if self.training_params['enable_distill_penalty'] and total_idx > divide_line:
                    soft_label = torch.abs(fake_data - X_fed[total_idx - data.shape[0] - divide_line:total_idx - divide_line, :]).sum()
                    g_loss += soft_label * self.training_params['lmd']

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

        # Save the trained parameters
        self.save_model()

    def generate_representation(self, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)

        z = Variable(torch.randn(self.num_samples, self.g_input).to(self.device))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples
        return self.to_np(samples)

    def to_np(self, X):
        return X.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), 'models/lrd/generator.pkl')
        torch.save(self.D.state_dict(), 'models/lrd/discriminator.pkl')
        print('Models save to models/lrd/generator.pkl & models/lrd/discriminator.pkl')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))
    
    @property
    def name(self):
        return 'GAN'

if __name__ == '__main__':
    gan = GAN(1024, 2048)
    X_task = torch.normal(0, 1, size=(1000, 1024))
    X_fed = torch.normal(0, 1, size=(1000, 1024))
    params = {
        'batch_size': 100,
        'num_epochs': 5,
        'd_LR': 0.0002,
        'd_WD': 0.00001,
        'g_LR': 0.0002,
        'g_WD': 0.00001,
        'enable_distill_penalty': True,
        'lmd': 0.0001
    }
    gan.train(X_task, X_fed, params)
    gan.generate_representation('./discriminator.pkl', './generator.pkl')