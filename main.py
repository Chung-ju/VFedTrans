import argparse
import time
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic')
    parser.add_argument('--type', type=str, default='intra')
    parser.add_argument('--frl', type=str, default='FedSVD')
    parser.add_argument('--lkt', type=str, default='AE')
    parser.add_argument('--task_model', type=str, default='xgboost')
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--gpu', '-g', type=str, default='0', help='Choose your GPU resource.')
    parser.add_argument('--seed', type=int, default=500)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    device = get_gpu(args.gpu)
    
    tp_data_dict = np.load(f'dataset/{args.dataset}/tp_{args.type}.npy', allow_pickle=True).item()
    dp_data_dict = np.load(f'dataset/{args.dataset}/dp_{args.type}.npy', allow_pickle=True).item()
    
    # Federated Representation Learning
    if os.path.exists(f'dataset/{args.dataset}/{args.frl}_{args.type}.npy'):
        fed_dict = np.load(f'dataset/{args.dataset}/{args.frl}_{args.type}.npy', allow_pickle=True).item()
    else:
        frl_trainer = FRL_Trainer()
        start_time = time.time()
        Xs_fed = frl_trainer.train(model=args.frl, Xs=[tp_data_dict['ol_sample'].astype(np.float64), dp_data_dict['ol_sample'].astype(np.float64)])
        end_time = time.time()
        fed_dict = {'Xs_fed': Xs_fed}
        np.save(f'dataset/{args.dataset}/{args.frl}_{args.type}.npy', fed_dict)
        print('FRL time: {:.4f}s'.format(end_time - start_time))
    print('Federated Representation Learning is done.')

    # Local Knowledge Transfer
    if os.path.exists(f'models/{args.dataset}_{args.lkt}_{args.type}.pt'):
        lkt_trainer = LKT_Trainer()
        lkt_trainer.set_model(args.lkt, input_dim=tp_data_dict['nl_sample'].shape[1], latent_dim=args.latent_dim)
        lkt_trainer.load_lkt_model(f'models/{args.dataset}_{args.lkt}_{args.type}.pt')
        lkt_trainer.load_dt_model(fed_dict['Xs_fed'].shape[1], args.latent_dim, f'models/{args.dataset}_{args.lkt}_{args.type}_dt.pt')
    else:
        lkt_trainer = LKT_Trainer()
        lkt_trainer.train(args.lkt, tp_data_dict['nl_sample'].reshape((tp_data_dict['nl_sample'].shape[0], -1)), 
                          fed_dict['Xs_fed'], args.latent_dim, device)
        lkt_trainer.save_model(f'models/{args.dataset}_{args.lkt}_{args.type}.pt', f'models/{args.dataset}_{args.lkt}_{args.type}_dt.pt')
    print('Local Knowledge Transfer is done.')
    
    # Feature augmentation
    X_aug = lkt_trainer.augment_feature(args.dataset, args.lkt, tp_data_dict['nl_sample'].reshape((tp_data_dict['nl_sample'].shape[0], -1)), device)
    task_trainer = DownstreamTask(train_ratio=0.1, random_state=args.seed)
    if args.dataset == 'leukemia' or args.dataset == 'pneumonia':
        task_trainer.run_img(tp_data_dict['nl_sample'], X_aug, tp_data_dict['nl_label'], args.task_model, device)
    else:
        task_trainer.run(X_aug, tp_data_dict['nl_label'], args.dataset, args.task_model, device, search=True)
        task_trainer.run(tp_data_dict['nl_sample'], tp_data_dict['nl_label'], args.dataset, args.task_model, device, search=False)
    