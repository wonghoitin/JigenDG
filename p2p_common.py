import requests
import yaml
import torch
import os
import numpy as np
from configs.default import p2p_models_path, versions_path


def send_request(node_ip, port, initiator_version):
    loading = {"initiator_version": initiator_version}
    r = requests.post(url="http://{}:{}/receive_ping".format(node_ip, port), params=loading)
    data = r.json()
    return data


def save_latest_params(latest_version, latest_params):
    with open("node_params.yaml", "w") as stream:
        yaml.dump({"latest_version": latest_version, "latest_params": latest_params}, stream)
        stream.close()
    return True


def save_latest_params_single(node, domain, latest_version):
    with open(versions_path + os.sep + "{}_{}_node_params.yaml".format(node, domain), "w") as stream:
        yaml.dump({"node_name": node, "domain": domain, "latest_version": latest_version}, stream)
        stream.close()
    return True


def check_version_single(node, domain, initiator_version):
    with open(versions_path + os.sep + "{}_{}_node_params.yaml".format(node, domain), "r") as stream:
        version = yaml.safe_load(stream)["latest_version"]
        stream.close()
    return version >= initiator_version


def save_state_dict(args, model, epochs, path, node, domain, optimizer=None, schedule=None):
    check_dict = {'args': args, 'epochs': epochs, 'model': model.state_dict()}
    if optimizer is not None:
        check_dict['optimizer'] = optimizer.state_dict()
    if schedule is not None:
        check_dict['shceduler'] = schedule.state_dict()
    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save(check_dict, os.path.join(path, "{}_{}_state_dict".format(node, domain) + '.pt'))


def load_torch_model(path):
    model = torch.load(path)
    return model


def read_node_list():
    with open(versions_path + os.sep + "node_list.yaml") as stream:
        nodes = yaml.safe_load(stream)
        stream.close()
    return nodes["nodes"]


def simple_ensemble(nodes, weights, test_data, metric):
    models = []
    for node, node_content in nodes.items():
        models.append(
            load_torch_model(os.path.join(p2p_models_path, "{}_{}_state_dict".format(node, node_content[2]) + '.pt')))

    with torch.no_grad():
        for imgs, labels, domain_labels, in test_data:
            imgs = imgs.cuda()
            votes = []
            for model in models:
                votes.append(model(imgs))
            votes = np.array(votes)
            output = weights.dot(votes)
            metric.update(output, labels)

    results_dict = metric.results()

    return results_dict