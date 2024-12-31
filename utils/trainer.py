import yaml
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.autograd import Variable

from .loss import *

# from utils import *
# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

import sys
sys.path.append('..')
from models.frl import *
from models.lkt import *

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

class FRL_Trainer():
    def __init__(self) -> None:
        super(FRL_Trainer, self).__init__()
    
    def train(self, **kwargs) -> np.ndarray:
        if kwargs['model'] == 'FedSVD':
            model = FedSVD()
            model.learning(kwargs['Xs'])
            Xs_fed = model.get_fed_representation()
        elif kwargs['model'] == 'VFedPCA':
            model = VFedPCA()
            model_config = load_model_config(kwargs['model'])
            X_task, X_data = kwargs['Xs'][0], kwargs['Xs'][1]
            X_full = np.concatenate([X_task, X_data], axis=1)
            Xs_fed = model.fed_representation_learning(model_config, X_full, [X_task, X_data])
        
        return Xs_fed
    
class LKT_Trainer():
    def __init__(self) -> None:
        super(LKT_Trainer, self).__init__()
        
    def set_model(self, model, input_dim: int, latent_dim: int) -> None:
        self.lkt_model = lrd_models[model](input_dim, latent_dim)
                
    def train_ae(self, 
                 X_nl, X_fed, 
                 latent_dim, device, 
                 batch_size=64, 
                 shuffle=False, 
                 num_epochs=30, 
                 lr=1e-3, 
                 beta_0=1e-3, 
                 beta_1=1e-1):
        nl_loader = torch.utils.data.DataLoader(
            dataset=torch.from_numpy(X_nl).to(device).to(torch.float32),
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        X_fed = torch.from_numpy(X_fed).to(device).to(torch.float32)
        self.lkt_model = self.lkt_model.to(device)
        T_theta = TNetwork(h_dim=latent_dim, z_dim=latent_dim).to(device)
        
        mse_loss = nn.MSELoss()
        self.domain_trans = DomainLearner(X_fed.shape[1], latent_dim).to(device)
        
        model_optm = optim.Adam(self.lkt_model.parameters(), lr=lr)
        T_optm = optim.Adam(T_theta.parameters(), lr=1e-4)
        dt_optm = optim.Adam(self.domain_trans.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            with tqdm(nl_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
                for batch in t:
                    enc, dec = self.lkt_model(batch)
                    
                    # Reconstruction loss
                    recons_loss = mse_loss(batch, dec)
                    
                    # Cross-domain invariant feature
                    z_nl = self.domain_trans(enc, X_fed)
                    
                    # Mutual information loss
                    z_nl_shuffled = sample_marginal(z_nl, z_nl.shape[0])
                    T_joint = T_theta(enc, z_nl)
                    T_marginal = T_theta(enc, z_nl_shuffled)
                    mi_loss = -(T_joint.mean() - torch.log(torch.exp(T_marginal).mean()))
                    
                    # Total loss
                    loss = beta_0 * recons_loss + beta_1 * mi_loss
                    total_loss += loss
                    
                    # t.set_postfix(loss=total_loss.item())
                
                # print('Reconstruction loss:', beta_0 * recons_loss)
                # print('Mutual information loss:', beta_1 * mi_loss)    
                    
                model_optm.zero_grad()
                dt_optm.zero_grad()
                T_optm.zero_grad()
                total_loss.backward()
                model_optm.step()
                dt_optm.step()
                T_optm.step()

    def train_vae(self,
                  X_nl, X_fed, 
                  latent_dim, device, 
                  batch_size=64, 
                  shuffle=False, 
                  num_epochs=30, 
                  model_lr=1e-3,
                  T_lr=1e-4,
                  dt_lr=1e-3,
                  beta_0=1e-3, 
                  beta_1=1e-1
                  ):
        nl_loader = torch.utils.data.DataLoader(
            dataset=torch.from_numpy(X_nl).to(device).to(torch.float32),
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        X_fed = torch.from_numpy(X_fed).to(device).to(torch.float32)
        self.lkt_model = self.lkt_model.to(device)
        T_theta = TNetwork(h_dim=latent_dim, z_dim=latent_dim).to(device)
        
        self.domain_trans = DomainLearner(X_fed.shape[1], latent_dim).to(device)
        
        model_optm = optim.Adam(self.lkt_model.parameters(), lr=model_lr)
        T_optm = optim.Adam(T_theta.parameters(), lr=T_lr)
        dt_optm = optim.Adam(self.domain_trans.parameters(), lr=dt_lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            with tqdm(nl_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
                for batch in t:
                    enc, recon_x, mean, logvar = self.lkt_model(batch)
                    recon_loss = vae_loss(recon_x, batch, mean, logvar)
                    
                    # Mutual information loss
                    z_nl = self.domain_trans(enc, X_fed)
                    z_nl_shuffled = sample_marginal(z_nl, z_nl.shape[0])
                    T_joint = T_theta(enc, z_nl)
                    T_marginal = T_theta(enc, z_nl_shuffled)
                    mi_loss = -(T_joint.mean() - torch.log(torch.exp(T_marginal).mean()))
                    
                    total_loss = beta_0 * recon_loss + beta_1 * mi_loss
                
                model_optm.zero_grad()
                dt_optm.zero_grad()
                T_optm.zero_grad()
                total_loss.backward()
                model_optm.step()
                dt_optm.step()
                T_optm.step()
                
    def train_wae(self,
                  X_nl, X_fed, 
                  latent_dim, device, 
                  batch_size=64, 
                  shuffle=False, 
                  num_epochs=30, 
                  model_lr=1e-3,
                  T_lr=1e-4,
                  dt_lr=1e-3,
                  disc_lr=1e-3,
                  beta_0=1e-3, 
                  beta_1=1e-1
                  ):
        nl_loader = torch.utils.data.DataLoader(
            dataset=torch.from_numpy(X_nl).to(device).to(torch.float32),
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        X_fed = torch.from_numpy(X_fed).to(device).to(torch.float32)
        self.lkt_model = WAE(X_nl.shape[1], latent_dim).to(device)
        T_theta = TNetwork(h_dim=latent_dim, z_dim=latent_dim).to(device)
        discriminator = Discriminator(latent_dim).to(device)
        self.domain_trans = DomainLearner(X_fed.shape[1], latent_dim).to(device)
        
        model_optm = optim.Adam(self.lkt_model.parameters(), lr=model_lr)
        T_optm = optim.Adam(T_theta.parameters(), lr=T_lr)
        dt_optm = optim.Adam(self.domain_trans.parameters(), lr=dt_lr)
        disc_optm = optim.Adam(discriminator.parameters(), lr=disc_lr)
        
        for epoch in range(num_epochs):
            total_loss = 0
            with tqdm(nl_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:                
                for batch in t:
                    enc, recon_x, _, _ = self.lkt_model(batch)
                    disc_optm.zero_grad()
                    real_loss = -torch.mean(discriminator(enc))
                    fake_loss = torch.mean(discriminator(torch.randn_like(enc)))
                    disc_loss = real_loss + fake_loss
                    disc_loss.backward(retain_graph=True)
                    disc_optm.step()
                    
                    recon_loss = wae_loss(recon_x, batch, discriminator, enc)
                    
                    # Mutual information loss
                    z_nl = self.domain_trans(enc, X_fed)
                    z_nl_shuffled = sample_marginal(z_nl, z_nl.shape[0])
                    T_joint = T_theta(enc, z_nl)
                    T_marginal = T_theta(enc, z_nl_shuffled)
                    mi_loss = -(T_joint.mean() - torch.log(torch.exp(T_marginal).mean()))
                    
                    total_loss = beta_0 * recon_loss + beta_1 * mi_loss
                
                model_optm.zero_grad()
                dt_optm.zero_grad()
                T_optm.zero_grad()
                total_loss.backward()
                model_optm.step()
                dt_optm.step()
                T_optm.step()
    
    def finetune(self, 
                 X_nl, X_fed, 
                 X_fed_list, dt_list, 
                 device, 
                 batch_size=64, 
                 shuffle=False, 
                 num_epochs=30, 
                 lr=1e-3):
        nl_loader = torch.utils.data.DataLoader(
            dataset=torch.from_numpy(X_nl).to(device).to(torch.float32),
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        X_fed = torch.from_numpy(X_fed).to(device).to(torch.float32)
        X_fed_list = [torch.from_numpy(X).to(device).to(torch.float32) for X in X_fed_list]
        
        self.lkt_model = self.lkt_model.to(device)
        dt_list = [dt.to(device) for dt in dt_list]
        self.domain_trans = self.domain_trans.to(device)
        
        model_optm = optim.Adam(self.lkt_model.parameters(), lr=lr)
        
        for i in range(len(dt_list)):
            dt_list[i].eval()
        
        for epoch in range(num_epochs):
            total_loss = 0
            with tqdm(nl_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as t:
                for batch in t:
                    enc, _ = self.lkt_model(batch)
                    
                    z_nl = self.domain_trans(enc, X_fed)
                    z_nl_list = []
                    z_nl_list.append(dt_list[i](enc, X_fed_list[i]))
                    
                    loss = compute_infonce_loss(enc, z_nl, z_nl_list)
                    total_loss += loss
                    
                model_optm.zero_grad()
                total_loss.backward()
                model_optm.step()
    
    def train(self, model, X_nl, X_fed, latent_dim, device):
        self.set_model(model, X_nl.shape[1], latent_dim)
        if model == 'AE':
            self.train_ae(X_nl, X_fed, latent_dim, device, batch_size=64, shuffle=False, num_epochs=20, lr=1e-3, beta_0=1e-3, beta_1=1e-1)
        elif model == 'VAE':
            self.train_vae(X_nl, X_fed, latent_dim, device)
        elif model == 'WAE':
            self.train_wae(X_nl, X_fed, latent_dim, device)
    
    def save_model(self, path, path1):
        torch.save(self.lkt_model.state_dict(), path)
        torch.save(self.domain_trans.state_dict(), path1)
    
    def load_lkt_model(self, path):
        self.lkt_model.load_state_dict(torch.load(path))
    
    def load_dt_model(self, input_dim, latent_dim, path):
        self.domain_trans = DomainLearner(input_dim, latent_dim)
        self.domain_trans.load_state_dict(torch.load(path))
    
    def augment_feature(self, dataset, model, original_feature, device):
        original_feature = torch.from_numpy(original_feature).to(device).to(torch.float32)
        self.lkt_model = self.lkt_model.to(device)
        if model == 'AE':
            augmented_feature, _ = self.lkt_model(original_feature)
        elif model == 'VAE':
            augmented_feature, _, _, _ = self.lkt_model(original_feature)
        elif model == 'WAE':
            augmented_feature, _, _, _ = self.lkt_model(original_feature)
        
        if dataset == 'leukemia' or dataset == 'pneumonia':
            final_feature = augmented_feature.cpu().detach().numpy()
        else:
            final_feature = np.concatenate([original_feature.cpu().detach().numpy(), augmented_feature.cpu().detach().numpy()], axis=1)
        
        return final_feature
        