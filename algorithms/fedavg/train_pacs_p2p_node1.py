import os
import argparse
from network.get_network import GetNetwork
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from utils.classification_metric import Classification
from utils.log_utils import *
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate
from utils.trainval_func import site_evaluation, site_train, GetFedModel, SaveCheckPoint
import p2p_common
from configs.default import p2p_models_path
import torch.nn.functional as F
from tqdm import tqdm
from kd import knowledge_distillation
from utils.weight_adjust import refine_weight_dict_by_GA


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'], help='model name')
    parser.add_argument("--test_domain", type=str, default='p',
                        choices=['p', 'a', 'c', 's'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=7)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--comm', help='epochs number', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='fedavg')
    parser.add_argument('--display', help='display in controller', action='store_true')
    parser.add_argument('--do_kd', help='To do knowledge distillation', type=bool, default=False)
    parser.add_argument('--do_ga', help='To do generalization adjustment', type=bool, default=False)
    parser.add_argument('--step_size', help='rate weight step', type=float, default=0.2)
    return parser.parse_args()


def main():
    nodes = p2p_common.read_node_list()

    node_name = "node1"
    self_domain = "a"

    '''log part'''
    file_name = 'fedavg_' + os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)

    '''dataset and dataloader'''
    dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    dataloader_dict, dataset_dict = dataobj.GetData()

    '''model'''
    metric = Classification()
    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)
    weight_dict = Cal_Weight_Dict(dataset_dict, site_list=dataobj.train_domain_list)
    FedUpdate(model_dict, global_model)
    site_results_before_avg = {}
    site_results_after_avg = {}

    for site_name in dataobj.train_domain_list:
        weight_dict[site_name] = 1. / 3.
        site_results_before_avg[site_name] = None
        site_results_after_avg[site_name] = None

    best_val = 0.
    step_size_decay = args.step_size / args.comm
    for i in range(args.comm + 1):
        FedUpdate(model_dict, global_model)

        site_train(i, self_domain, args, model_dict[self_domain], optimizer_dict[self_domain],
                   scheduler_dict[self_domain], dataloader_dict[self_domain]['train'], log_ten, metric)

        site_results_before_avg[self_domain] = site_evaluation(i, self_domain, args, model_dict[self_domain],
                                                               dataloader_dict[self_domain]['val'],
                                                               log_file, log_ten, metric, note='before_fed')

        p2p_common.save_latest_params_single(node_name, self_domain, i)
        p2p_common.save_state_dict(args, model_dict[self_domain], i, p2p_models_path, node_name, self_domain,
                                   optimizer_dict[self_domain], scheduler_dict[self_domain])

        if i != 0:
            for node_index in range(len(nodes)):
                if "node{}".format(node_index + 1) != node_name:
                    # if p2p_common.check_version_single(node, node_content[2], i):
                    loaded_dict = p2p_common.load_torch_model(
                        p2p_models_path + "\\{}_{}_state_dict.pt".format("node{}".format(node_index + 1),
                                                                         nodes[node_index][
                                                                             "node{}".format(node_index + 1)][2]))
                    model_dict[nodes[node_index]["node{}".format(node_index + 1)][2]].load_state_dict(
                        loaded_dict["model"])

        FedAvg(model_dict, weight_dict, global_model)

        if args.do_kd:
            for kd_epoch in range(args.kd_epochs):
                knowledge_distillation(model_dict[self_domain], global_model, dataloader_dict[self_domain]['train'],
                                       optimizer_dict[self_domain], metric, args)

        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            site_results_after_avg[domain_name] = site_evaluation(i, domain_name, args, global_model,
                                                                  dataloader_dict[domain_name]['val'],
                                                                  log_file, log_ten, metric)
            fed_val += site_results_after_avg[domain_name]['acc'] * weight_dict[domain_name]
        # val 结果
        if fed_val >= best_val:
            best_val = fed_val
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            for domain_name in dataobj.train_domain_list:
                SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'),
                               note=f'best_val_{domain_name}_model')

            log_file.info(f'Model saved! Best Val Acc: {best_val * 100:.2f}%')
        site_evaluation(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], log_file,
                        log_ten, metric, note='test_domain')

        if args.do_ga:
            weight_dict = refine_weight_dict_by_GA(weight_dict, site_results_before_avg, site_results_after_avg,
                                                   args.step_size - (i - 1) * step_size_decay, fair_metric=args.fair)

    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list:
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'),
                       note=f'last_{domain_name}_model')


if __name__ == '__main__':
    main()
