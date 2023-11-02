import torch
import torch.nn as nn
import torch.nn.functional as F


def knowledge_distillation(local_model, global_model, dataloader, optimzier, metric, params):
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        optimzier.zero_grad()
        local_output = local_model(imgs)
        teacher_output = global_model(imgs)
        loss = get_kd_loss(local_output, teacher_output, labels, params)
        loss.backward()
        optimzier.step()


def get_kd_loss(local_output, teacher_output, labels, params):
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(local_output / T, dim=1),
                             F.softmax(teacher_output / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(local_output, labels) * (1. - alpha)

    return KD_loss
