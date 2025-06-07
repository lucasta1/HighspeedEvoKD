import random
import torch
from nas_201_api import NASBench201API
from models.nasbench201.utils import dict2config, get_cell_based_tiny_net


# Load API safely
file_path = '/home/taichi/Project_KU2/HighspeedEvoKD/data/NAS-Bench-201-v1_1-096897.pth'
# file_obj = torch.load(file_path, map_location='cpu', weights_only=False)
nb201_api = NASBench201API(
    file_path_or_dict=file_path, verbose=False)


def get_nb201_api():
    """
    Returns the NASBench201 API instance.
    """
    return nb201_api


def get_nb201_model(choiced_index):
    arch_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': nb201_api.arch(choiced_index),
        'num_classes': 100
    }
    net_config = dict2config(arch_config, None)
    model = get_cell_based_tiny_net(net_config)
    return model


def get_rnd_nb201_and_acc(dataset='cifar100'):
    index = random.randint(0, len(nb201_api) - 1)
    model = get_nb201_model(index)

    # 利用可能な試行を取得
    results = nb201_api.query_by_index(index, dataset)
    available_seeds = list(results.keys())
    chosen_seed = random.choice(available_seeds)

    acc = results[chosen_seed].get_eval('x-test')['accuracy']
    return model, acc, index

