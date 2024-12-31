import argparse
import warnings
warnings.filterwarnings("ignore")

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic')
    parser.add_argument('--type', type=str, default='intra')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=500)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    partition_params = load_partition_params(args.dataset)
    
    if args.dataset == 'leukemia' or args.dataset == 'pneumonia':
        img_partition(args.dataset, partition_params, shuffle=args.shuffle)
    else:
        if args.type == 'intra':
            partition(args.dataset, partition_params, partition_type=args.type, shuffle=args.shuffle)
        else:
            noniid_partition(args.dataset, partition_params, shuffle=args.shuffle)
    
    # tp_data_dict = np.load('dataset/credit/tp.npy', allow_pickle=True).item()
    # ol_tp_data = VFedDataset(type='val', X=tp_data_dict['ol_sample'], y=tp_data_dict['ol_label'], ids=tp_data_dict['ol_ids'])
    # print(ol_tp_data[0])