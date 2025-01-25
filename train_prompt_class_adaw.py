from __future__ import print_function
import os
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
# from advertorch.attacks import LinfPGDAttack
from torchattacks import PGD
from torch.utils.data import Dataset
import torch.nn as nn
import time
import datetime
from torchvision.utils import save_image
from collections import defaultdict
from resnet import *
# from WRN_cifar import *


parser = argparse.ArgumentParser(description='Prompt Defense')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./result/newAT(ori_sort)',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--batch_size', type=int, default=512, metavar='B',
                    help='batch_size')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch):
    """decrease lr"""
    lr = args.lr
    if epoch >= 90:
        lr = args.lr * 0.001
    elif epoch >= 60:
        lr = args.lr * 0.01
    elif epoch >= 30:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, model, device, train_loader, epoch, prompt_set_abs, prompt_set_ang, lambda_abs):

    for batch_idx, (data, label) in enumerate(train_loader):

        data, label = data.to(device), label.to(device)
        data_adv = craft_adv_train(model=model, x_natural=data, y=label)

        fft_adv = torch.fft.fftn(data_adv, dim=(1, 2, 3))
        abs_adv, angle_adv = torch.abs(fft_adv), torch.angle(fft_adv)
        fft_adv_rec = (abs_adv+lambda_abs*prompt_set_abs[label.tolist()]) * torch.exp((1j) * (angle_adv+1*prompt_set_ang[label.tolist()]))
        data_adv_rec = torch.fft.ifftn(fft_adv_rec, dim=(1, 2, 3)).float()

        fft_nat = torch.fft.fftn(data, dim=(1, 2, 3))
        abs_nat, angle_nat = torch.abs(fft_nat), torch.angle(fft_nat)
        fft_nat_rec = (abs_nat + lambda_abs * prompt_set_abs[label.tolist()]) * torch.exp(
            (1j) * (angle_nat + 1 * prompt_set_ang[label.tolist()]))
        data_nat_rec = torch.fft.ifftn(fft_nat_rec, dim=(1, 2, 3)).float()
        logits_nat_rec = model(data_nat_rec)
        loss_ce_nat_rec = F.cross_entropy(logits_nat_rec, label)

        logits_adv_rec = model(data_adv_rec)
        loss_ce_adv_rec = F.cross_entropy(logits_adv_rec, label)
        loss_mse = torch.mean(torch.exp(torch.abs(data_adv_rec - data)))
        loss_cw = loss_cw_(data_adv, label, prompt_set_abs, prompt_set_ang, model, lambda_abs)
        loss = loss_ce_adv_rec + 5000 * loss_mse + 4 * loss_cw + 1 * loss_ce_nat_rec

        lr=0.1
        if epoch>=75:
            lr=0.01
        grad = torch.autograd.grad(loss, prompt_set_abs, retain_graph=True)[0]
        prompt_set_abs = prompt_set_abs - lr * grad.sign()

        grad = torch.autograd.grad(loss, prompt_set_ang, retain_graph=True)[0]
        prompt_set_ang = prompt_set_ang - lr * grad.sign()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_ce_adv_rec: {:.6f}, Loss_ce_nat_rec: {:.6f}, Loss_mse: {:.6f}, Loss_cw: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss_ce_adv_rec.item(), loss_ce_nat_rec.item(), loss_mse.item(), loss_cw.item()))
    return prompt_set_abs, prompt_set_ang

def loss_cw_(data, label, prompt_set_abs, prompt_set_ang, model, lambda_abs):

    min_label, max_label = label.min(), label.max()

    random_labels = torch.randint(min_label, max_label + 1, label.shape, device=label.device)

    same_position = random_labels == label

    if same_position.any():

        alternative_labels = torch.randint(min_label, max_label, label.shape, device=label.device)
        random_labels[same_position] = torch.where(alternative_labels[same_position] >= label[same_position],
                                                   alternative_labels[same_position] + 1,
                                                   alternative_labels[same_position])

    fft = torch.fft.fftn(data, dim=(1, 2, 3))
    abs, angle = torch.abs(fft), torch.angle(fft)
    fft_rec = (abs + lambda_abs * prompt_set_abs[random_labels.tolist()]) * torch.exp(
        (1j) * (angle + prompt_set_ang[random_labels.tolist()]))
    data_rand_rec = torch.fft.ifftn(fft_rec, dim=(1, 2, 3)).float()


    logits_rand_rec = model(data_rand_rec)
    logits_selected = torch.gather(logits_rand_rec, dim=1, index=random_labels.unsqueeze(1))
    logits_selected_true = torch.gather(logits_rand_rec, dim=1, index=label.unsqueeze(1))


    logits_diff = logits_selected - logits_selected_true
    loss = torch.max(logits_diff, -0.1 * torch.ones_like(logits_diff))
    loss = loss.mean()

    return loss

def craft_adv_train(model, x_natural, y):
    # attack = LinfPGDAttack(model, eps=8/255, nb_iter=10, eps_iter=2/255, targeted=False)
    attack = PGD(model, eps=8/255, alpha=2/255, steps=10)
    x_adv = attack(x_natural, y)
    return x_adv

def craft_adv_test(model, x_natural, y):
    # attack = LinfPGDAttack(model, eps=8/255, nb_iter=10, eps_iter=2/255, targeted=False)
    attack = PGD(model, eps=8/255, alpha=2/255, steps=10)
    x_adv = attack(x_natural, y)
    return x_adv

def rec_prompt(prompt_set_abs, prompt_set_ang, pre_label, data, lambda_abs):

    fft_samples = torch.fft.fftn(data, dim=(1, 2, 3))
    abs_samples, angle_samples = torch.abs(fft_samples), torch.angle(fft_samples)
    fft_rec = (abs_samples+lambda_abs*prompt_set_abs[pre_label.tolist()]) * torch.exp((1j) * (angle_samples+1*prompt_set_ang[pre_label.tolist()]))
    modified_samples = torch.fft.ifftn(fft_rec, dim=(1, 2, 3)).float()

    return modified_samples

def eval_predict(model, data, label):
    logits = model(data)
    pred = logits.max(1, keepdim=True)[1]
    return pred.eq(label.view_as(pred)).sum().item()

def eval_test(model, device, test_loader, prompt_set_abs, prompt_set_ang, epoch, lambda_abs):
    correct = 0
    correct_adv = 0
    correct_rec_adv = 0
    correct_rec_nat = 0

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        data_adv = craft_adv_test(model=model, x_natural=data, y=label)

        correct += eval_predict(model, data, label)
        correct_adv += eval_predict(model, data_adv, label)

        _, pre_label_nat = torch.max(model(data), dim=1)
        _, pre_label_adv = torch.max(model(data_adv), dim=1)

        data_rec_nat = rec_prompt(prompt_set_abs, prompt_set_ang, pre_label_nat, data, lambda_abs)
        data_rec_adv = rec_prompt(prompt_set_abs, prompt_set_ang, pre_label_adv, data_adv, lambda_abs)

        correct_rec_nat += eval_predict(model, data_rec_nat, label)
        correct_rec_adv += eval_predict(model, data_rec_adv, label)

    print('Test: Nat Acc: {}/{} ({:.5f}%), Rob Acc: {}/{} ({:.5f}%), Rec Nat Acc: {}/{} ({:.5f}%), Rec Rob Acc: {}/{} ({:.5f}%)'.format(
        correct,     len(test_loader.dataset), 100. * correct     / len(test_loader.dataset),
        correct_adv, len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset),
        correct_rec_nat, len(test_loader.dataset), 100. * correct_rec_nat / len(test_loader.dataset),
        correct_rec_adv, len(test_loader.dataset), 100. * correct_rec_adv / len(test_loader.dataset)))


def ada_acc_train(model, device, train_loader, prompt_set_abs, prompt_set_ang, lambda_abs):

    correct_rec_adv_abs = 0
    correct_rec_adv_ang = 0

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

        data_adv = craft_adv_train(model=model, x_natural=data, y=label)

        fft_adv = torch.fft.fftn(data_adv, dim=(1, 2, 3))
        abs_adv, angle_adv = torch.abs(fft_adv), torch.angle(fft_adv)

        fft_rec_abs = (abs_adv + lambda_abs*prompt_set_abs[label.tolist()]) * torch.exp((1j) * angle_adv)
        fft_rec_ang = abs_adv * torch.exp((1j) * (angle_adv + 1*prompt_set_ang[label.tolist()]))

        data_rec_adv_abs = torch.fft.ifftn(fft_rec_abs, dim=(1, 2, 3)).float()
        data_rec_adv_ang = torch.fft.ifftn(fft_rec_ang, dim=(1, 2, 3)).float()

        correct_rec_adv_abs += eval_predict(model, data_rec_adv_abs, label)
        correct_rec_adv_ang += eval_predict(model, data_rec_adv_ang, label)

    print('Test: Rec adv abs Acc: {}/{} ({:.5f}%), Rec adv ang Acc: {}/{} ({:.5f}%)'.format(
        correct_rec_adv_abs, len(train_loader.dataset), 100. * correct_rec_adv_abs / len(train_loader.dataset),
        correct_rec_adv_ang, len(train_loader.dataset), 100. * correct_rec_adv_ang / len(train_loader.dataset)))
    print('lambda_abs*(correct_rec_adv_abs/correct_rec_adv_ang)={}'.format(lambda_abs*(correct_rec_adv_abs/correct_rec_adv_ang)))

    return lambda_abs * (correct_rec_adv_abs / correct_rec_adv_ang)


def main():
    # settings
    setup_seed(args.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")


    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    ''' =============== setup data loader =================== '''
    train_dataset = datasets.CIFAR10(os.path.expanduser('./data/'), train=True, download=True, transform=trans_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=4, shuffle=True,
                                               drop_last=False, pin_memory=True)

    test_dataset = datasets.CIFAR10(os.path.expanduser('./data/'), train=False, download=True, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=4, shuffle=False,
                                              drop_last=False, pin_memory=True)
    ''' =============== setup data loader =================== '''

    ''' ================= Prompt Set Construction ================== '''

    class_samples = defaultdict(list)


    for images, labels in train_loader:
        for i in range(len(labels)):
            class_samples[labels[i].item()].append(images[i].unsqueeze(0))


    one_sample_per_class = [class_samples[i][0] for i in range(10)]

    batch_samples = torch.cat(one_sample_per_class)
    print(batch_samples.shape)
    fft = torch.fft.fftn(batch_samples, dim=(1, 2, 3))
    abs, angle = torch.abs(fft), torch.angle(fft)

    prompt_set_abs = abs.to(device)
    prompt_set_abs.requires_grad = True
    prompt_set_ang = angle.to(device)
    prompt_set_ang.requires_grad = True
    ''' ================= Prompt Set Construction ================== '''


    # ================================== ResNet18 ================================
    model = ResNet18().to(device)
    # best_path = torch.load("./result/train_nat_cifar10/100_ori_res18.pth")
    best_path = torch.load("./result/train_AT_cifar10/90_Baseline_standAT_cifar10(resnet18).pth")
    # best_path = torch.load("./result/train_TR_cifar10/90_Baseline_TRADES_cifar10(resnet18).pth")
    # best_path = torch.load("./result/train_MA_cifar10/90_Baseline_MART_cifar10(resnet18).pth")

    model.load_state_dict(best_path)
    model = torch.nn.DataParallel(model)
    model.eval()

    cudnn.benchmark = True
    # optimizer_abs = optim.SGD([prompt_set_abs], lr=2 / 255, momentum=0.0, weight_decay=0)
    # optimizer_ang = optim.SGD([prompt_set_ang], lr=2 / 255, momentum=0.0, weight_decay=0)

    start_time = time.time()
    lambda_abs = 1
    for epoch in range(1, args.epochs + 1):
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        print('time:', et)

        # adjust_learning_rate(optimizer, epoch)
        prompt_set_abs, prompt_set_ang = train(args, model, device, train_loader, epoch, prompt_set_abs, prompt_set_ang, lambda_abs)


        if epoch % 5 == 0:
            eval_test(model, device, test_loader, prompt_set_abs, prompt_set_ang, epoch, lambda_abs)

        if epoch % 5 == 0:
            lambda_abs = ada_acc_train(model, device, train_loader, prompt_set_abs, prompt_set_ang, lambda_abs)

        # save prompt
        torch.save(prompt_set_abs, './result/prompt_abs_{}.pt'.format(epoch))
        torch.save(prompt_set_ang, './result/prompt_ang_{}.pt'.format(epoch))

        print('================================================================')

if __name__ == '__main__':
    main()
