import torch.nn.functional as F


def CE_loss(input_logits, target_targets, temperature=1):
    return F.cross_entropy(input=input_logits/temperature, target=target_targets)


def region_consistency_loss(x, y, cmask):
    loss_c = F.kl_div((x * cmask).softmax(dim=-1).log(), (y * cmask).softmax(dim=-1), reduction='mean')

    umask = 1 - cmask
    loss_u = F.kl_div((x * umask).softmax(dim=-1).log(), (y * umask).softmax(dim=-1), reduction='mean')

    total_pixels = cmask.numel()  # 计算总像素个数
    num_ones = (cmask == 1).sum()  # 计算像素值为 1 的个数
    num_zeroes = (cmask == 0).sum()  # 计算像素值为 0 的个数
    weight_ones = num_ones.float() / total_pixels  # 计算像素值为1的占比
    weight_zeroes = num_zeroes.float() / total_pixels  # 计算像素值为0的占比

    loss = weight_zeroes * loss_u + weight_ones * (1 - loss_c)
    return loss
