import argparse
import warnings
warnings.filterwarnings("ignore")

from models.frl import *
from models.lrd import *
from train_lrd import *
from train_frl import *
from performance import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a single task!')
    parser.add_argument('--gpu', '-g', type=str, default='0', help='Choose your GPU resource.')

    args = parser.parse_args()
    device = get_gpu(args.gpu)

    task_config = load_task_config()
    test = Performance()

    print('Dataset: ', task_config['dataset'])
    print('Split mode:', task_config['split'])
    print('Step 1 - FRL: ', task_config['frl'])
    print('Step 2 - LRD ', task_config['lrd'])
    print('Step 3 - Downstream method: ', task_config['method'])
    print('Run on ', device)

    # Split data
    X_task, y_task, X_shared, X_data = eval(task_config['split'] + '_split')(task_config['dataset'])
    print('Task hospital: ', X_task.shape)
    print('Data hospital: ', X_data.shape)
    print('Shared samples', X_shared.shape)

    # Step 1: Federated Representation Learning
    frl_model_config = load_model_config(task_config['frl'])
    frl_model = frl_models[task_config['frl']](num_features=X_shared.shape[1])
    frl = FedRepresentationLearning(frl_model, frl_model_config)
    Xs_fed = frl.training(X_task=X_task, X_data=X_data, X_shared=X_shared)
    print(Xs_fed.shape)

    # Step 2: Local Representation Distillation
    lrd_model_config = load_model_config(task_config['lrd'])
    lrd_model = lrd_models[task_config['lrd']](X_task.shape[1], Xs_fed.shape[1], **lrd_model_config['model_params'])
    lrd = LocalRepresentationDistillation(lrd_model, lrd_model_config['exp_params'], device)
    lrd.training_step(X_task, Xs_fed)

    # Step 3
    X_new = lrd.representation_distillation_step()
    print('-----Result-----')
    test.run(X_new, y_task, task_config['method'])