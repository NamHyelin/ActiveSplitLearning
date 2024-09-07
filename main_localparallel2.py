import torch
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import torchvision
from torchsummary import summary
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from model import resnet18, teacher_model, student_model, projection_head, feature_size, Decoder
from losses import DistillationLoss, SupConLoss, one_hot, mixup, get_cca_similarity
from .dataset import batchsize, SplitData, testloader, num_datasplits, new_dataload, NoniidSplitData
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import torch.nn.functional as F
import collections
import random
import time
import math
import copy
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# TODO Parameters
lr = 0.01
lr_c = 0.01
lr_decay = 0.95
num_clients = 5
alpha = 1
distribute_alpha=0.5  #작을수록 좋네(0.2) 더 갈라짐
Temperature = 7




# TODO model
server = teacher_model().to(device)
# client = student_model(Temperature).to(device)

client_models = [student_model(Temperature).to(device) for _ in range(num_clients)]
# best_model= student_model(Temperature).to(device)

pr_head= projection_head().to(device)
decoder= Decoder(feature_size[0], feature_size[1]).to(device)

optimizer_s = optim.SGD(server.parameters(), lr=lr, momentum=0.9)
# scheduler_s = optim.lr_scheduler.LambdaLR(optimizer=optimizer_s, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_ep och=-1, verbose=False)

# optimizer_c = [optim.SGD(model.parameters(), lr=lr_c, momentum=0.9) for model in client_models]
optimizer_c = [optim.Adam(model.parameters(), lr=lr_c) for model in client_models]
# scheduler_c = [optim.lr_scheduler.LambdaLR(optimizer=optimiz, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_epoch=-1, verbose=False) for optimiz in optimizer_c]

optimizer_p = optim.Adam(pr_head.parameters(), lr=lr)
optimizer_d = optim.Adam(decoder.parameters(), lr=lr)





# for i in range(num_clients):
#     torch.save(client_models[i].state_dict(), './initial_cli_{}.pth'.format(i))
# torch.save(server.state_dict(), './initial_ser.pth')



# TODO critierion
criterion_KL = DistillationLoss(distillation_type='soft', alpha=alpha, tau=Temperature)
criterion_CE = nn.CrossEntropyLoss()
criterion_CT = SupConLoss(contrast_mode='all', base_temperature=0.07)
criterion_BCE= nn.BCELoss()
criterion_MSE= nn.MSELoss()



# TODO dataset per clients

new_dataload(True)
clients_dataset = [SplitData(i, 0) for i in range(num_clients)]

# clients_dataset1 = [SplitData(i, 0) for i in range(int(num_clients/2))]
# clients_dataset2= [SplitData(i,1) for i in range(int(num_clients/2))]
# clients_dataset= clients_dataset1+clients_dataset2
# clients_dataset = clients_dataset1
# clients_dataloader = [DataLoader(dataset, batch_size=batchsize, shuffle=False) for dataset in clients_dataset]

# clients_dataset= NoniidSplitData(int(50000/num_clients), num_clients, distribute_alpha)
clients_dataloader = [DataLoader(dataset, batch_size=batchsize, shuffle=True) for dataset in clients_dataset]

for i in range(num_clients):
    y_train= [y for _,y in clients_dataset[i]]
    counter_train= collections.Counter(y_train)
    print(counter_train)




class Distill:
    def __init__(self, epoch):
        self.epoch = epoch

    def select_straggler(self, num_straggler):
        #random
        self.num_straggler = num_straggler
        straggler_list = random.sample(list(np.arange(num_clients)), self.num_straggler)
        client_list = [x for x in list(np.arange(num_clients)) if (x not in straggler_list)]

        was_cli[client_list]=1
        return client_list, straggler_list

    def select_straggler2(self, num_straggler, former_client_models, now_client_models, epoch):
        #uniformly
        if epoch%5==0:
            straggler_list= [0,1,2,3]
            client_list= [x for x in list(np.arange(num_clients)) if (x not in straggler_list)]
        elif epoch%5==1:
            straggler_list = [4,5,6,7]
            client_list = [x for x in list(np.arange(num_clients)) if (x not in straggler_list)]
        elif epoch%5==2:
            straggler_list = [8,9,0,1]
            client_list = [x for x in list(np.arange(num_clients)) if (x not in straggler_list)]
        elif epoch%5==3:
            straggler_list = [2,3,4,5]
            client_list = [x for x in list(np.arange(num_clients)) if (x not in straggler_list)]
        else:
            straggler_list = [6,7,8,9]
            client_list = [x for x in list(np.arange(num_clients)) if (x not in straggler_list)]

        return client_list, straggler_list


    def select_straggler3(self, num_straggler, former_client_models, now_client_models, datas):
        # l2 norm
        with torch.no_grad():
            if per_batch==True:
                norms=np.zeros((num_clients))
                for idx in range(num_clients):
                    input, label = datas[idx]
                    input, label = input.to(device), label.to(device)
                    now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                    act2, _, _ = now_client_models[idx](input)
                    former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                    act1, _, _ = former_client_models[idx](input)
                    norms[idx] += torch.norm((act1 - act2), 'fro')
                cli_norms= norms.argsort()
                straggler_list=cli_norms[-num_straggler:]
                client_list= cli_norms[:-num_straggler]
                was_cli[client_list] = 1

            else:
                norms = np.zeros((num_clients))
                for batch_idx, data in enumerate(zip(*clients_dataloader)):
                    for idx in range(num_clients):
                        input, label = data[idx]
                        input, label = input.to(device), label.to(device)
                        now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                        act2, _, _ = now_client_models[idx](input)
                        former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                        act1, _, _ = former_client_models[idx](input)
                        norms[idx] += torch.norm((act1 - act2), 'fro')
                cli_norms = norms.argsort()
                print(norms)
                straggler_list = cli_norms[:num_straggler] #[-num_straggler:]
                client_list = cli_norms[num_straggler:] #[:-num_straggler]
                was_cli[client_list] = 1
                f = open("D:/Dropbox/나메렝/wml/210827/scheduling/norm_cli5.txt", 'a')
                f.write(np.array2string(norms))
                f.write(np.array2string(straggler_list))
                f.close()

        return client_list, straggler_list


    def select_straggler4(self, num_straggler, former_client_models, now_client_models, datas):
        #Kullback-Leibler Divergence
        with torch.no_grad():
            if per_batch==True:
                norms=np.zeros((num_clients))
                for idx in range(num_clients):
                    input, label = datas[idx]
                    input, label = input.to(device), label.to(device)
                    former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                    _, act1, _ = former_client_models[idx](input)
                    now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                    _, act2, _ = now_client_models[idx](input)
                    norms[idx] += F.kl_div(F.log_softmax(act1 / 7, dim=1), F.log_softmax(act2 / 7, dim=1),
                                           reduction='sum', log_target=True)
                cli_norms= norms.argsort()
                straggler_list=cli_norms[-num_straggler:]
                client_list= cli_norms[:-num_straggler]
                was_cli[client_list] = 1

            else:
                norms = np.zeros((num_clients))
                for batch_idx, data in enumerate(zip(*clients_dataloader)):
                    for idx in range(num_clients):
                        input, label = data[idx]
                        input, label = input.to(device), label.to(device)
                        former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                        _, act1, _ = former_client_models[idx](input)
                        now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                        _, act2, _ = now_client_models[idx](input)
                        norms[idx] += F.kl_div(F.log_softmax(act1/7, dim=1), F.log_softmax(act2/7, dim=1), reduction='sum', log_target=True)
                cli_norms = norms.argsort()
                print(norms)
                straggler_list = cli_norms[:num_straggler] #[-num_straggler:]
                client_list = cli_norms[num_straggler:] #[:-num_straggler]
                was_cli[client_list] = 1
                f = open("D:/Dropbox/나메렝/wml/210827/scheduling/acc_cli5.txt", 'a')
                f.write(np.array2string(norms))
                f.write(np.array2string(straggler_list))
                f.close()

        return client_list, straggler_list




    def select_straggler5(self, num_straggler, former_client_models, now_client_models, datas):
        #Cosine loss
        with torch.no_grad():
            if per_batch==True:
                norms=np.zeros((num_clients))
                for idx in range(num_clients):
                    input, label = datas[idx]
                    input, label = input.to(device), label.to(device)
                    former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                    _, act1, _ = former_client_models[idx](input)
                    now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                    _, act2, _ = now_client_models[idx](input)
                    cos = torch.nn.CosineEmbeddingLoss()
                    norms[idx] += cos(act1, act2, label)
                cli_norms= norms.argsort()
                straggler_list=cli_norms[-num_straggler:]
                client_list= cli_norms[:-num_straggler]
                was_cli[client_list] = 1

            else:
                norms = np.zeros((num_clients))
                for batch_idx, data in enumerate(zip(*clients_dataloader)):
                    for idx in range(num_clients):
                        input, label = data[idx]
                        input, label = input.to(device), label.to(device)
                        former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                        _, _, act1 = former_client_models[idx](input)
                        now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                        _, _, act2 = now_client_models[idx](input)
                        cos = torch.nn.CosineEmbeddingLoss()
                        y= -torch.ones((128)).to(device)
                        norms[idx] += cos(act1, act2, y)
                cli_norms = norms.argsort()
                print(norms)
                straggler_list = cli_norms[:num_straggler] #[-num_straggler:]
                client_list = cli_norms[num_straggler:] #[:-num_straggler]
                was_cli[client_list] = 1
                f = open("D:/Dropbox/나메렝/wml/210827/scheduling/cosine_cli5.txt", 'a')
                f.write(np.array2string(norms))
                f.write(np.array2string(straggler_list))
                f.close()

        return client_list, straggler_list




    def select_straggler6(self, num_straggler, former_client_models, now_client_models, datas):
        #local acc
        with torch.no_grad():
            if per_batch==True:
                norms=np.zeros((num_clients))
                for idx in range(num_clients):
                    input, label = datas[idx]
                    input, label = input.to(device), label.to(device)
                    former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                    feat1, _, _ = former_client_models[idx](input)
                    now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                    feat2, _, _ = now_client_models[idx](input)
                    _, out1 = pr_head(feat1)
                    loss1 = criterion_CE(out1, label)
                    _, out2 = pr_head(feat2)
                    loss2 = criterion_CE(out2, label)
                    norms[idx] += abs(loss1 - loss2)
                cli_norms = norms.argsort()
                cli_norms= norms.argsort()
                straggler_list=cli_norms[-num_straggler:]
                client_list= cli_norms[:-num_straggler]
                was_cli[client_list] = 1

            else:
                norms = np.zeros((num_clients))
                for batch_idx, data in enumerate(zip(*clients_dataloader)):
                    for idx in range(num_clients):
                        input, label = data[idx]
                        input, label = input.to(device), label.to(device)
                        former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                        feat1, _, _ = former_client_models[idx](input)
                        now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                        feat2, _, _ = now_client_models[idx](input)
                        _, out1= pr_head(feat1)
                        loss1= criterion_CE(out1, label)
                        _, out2 = pr_head(feat2)
                        loss2 = criterion_CE(out2, label)
                        norms[idx] += abs(loss1-loss2)
                cli_norms = norms.argsort()
                print(norms)
                straggler_list = cli_norms[:num_straggler] #[-num_straggler:]
                client_list = cli_norms[num_straggler:] #[:-num_straggler]
                was_cli[client_list] = 1
                f = open("D:/Dropbox/나메렝/wml/210827/scheduling/acc_cli5.txt", 'a')
                f.write(np.array2string(norms))
                f.write(np.array2string(straggler_list))
                f.close()

        return client_list, straggler_list




    def select_straggler7(self, num_straggler, former_client_models, now_client_models, datas):
        #reconstruction loss
        with torch.no_grad():
            if per_batch==True:
                norms=np.zeros((num_clients))
                for idx in range(num_clients):
                    input, label = datas[idx]
                    input, label = input.to(device), label.to(device)
                    former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                    feat1, _, _ = former_client_models[idx](input)
                    now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                    feat2, _, _ = now_client_models[idx](input)
                    out1 = decoder(feat1)
                    loss1 = criterion_MSE(out1, input)
                    out2 = decoder(feat2)
                    loss2 = criterion_MSE(out2, input)
                    norms[idx] += abs(loss1 - loss2)
                cli_norms = norms.argsort()
                cli_norms= norms.argsort()
                straggler_list=cli_norms[-num_straggler:]
                client_list= cli_norms[:-num_straggler]
                was_cli[client_list] = 1

            else:
                norms = np.zeros((num_clients))
                for batch_idx, data in enumerate(zip(*clients_dataloader)):
                    for idx in range(num_clients):
                        input, label = data[idx]
                        input, label = input.to(device), label.to(device)
                        former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                        feat1, _, _ = former_client_models[idx](input)
                        now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                        feat2, _, _ = now_client_models[idx](input)
                        out1= decoder(feat1)
                        loss1= criterion_MSE(out1, input)
                        out2 = decoder(feat2)
                        loss2 = criterion_MSE(out2, input)
                        norms[idx] += abs(loss1-loss2)
                cli_norms = norms.argsort()
                print(norms)
                straggler_list = cli_norms[:num_straggler] #[-num_straggler:]
                client_list = cli_norms[num_straggler:] #[:-num_straggler]
                was_cli[client_list] = 1
                f = open("D:/Dropbox/나메렝/wml/210827/scheduling/reconst_cli5.txt", 'a')
                f.write(np.array2string(norms))
                f.write(np.array2string(straggler_list))
                f.close()

        return client_list, straggler_list




    def select_straggler8(self, num_straggler, former_client_models, now_client_models, datas):
        #SVCCA
        with torch.no_grad():
            if per_batch==True:
                norms=np.zeros((num_clients))
                for idx in range(num_clients):
                    input, label = datas[idx]
                    input, label = input.to(device), label.to(device)
                    former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                    feat1, _, _ = former_client_models[idx](input)
                    now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                    feat2, _, _ = now_client_models[idx](input)
                    bsz, channel, h, w = feat1.shape
                    act1 = feat1.reshape((bsz * h * w, bsz))
                    bsz, channel, h, w = feat2.shape
                    act2 = feat2.reshape((bsz * h * w, bsz))
                    f_results = get_cca_similarity(act1.T, act2.T, epsilon=1e-10, verbose=False)
                    norms[idx] += f_results["cca_coef1"]
                cli_norms = norms.argsort()
                cli_norms= norms.argsort()
                straggler_list=cli_norms[-num_straggler:]
                client_list= cli_norms[:-num_straggler]
                was_cli[client_list] = 1

            else:
                norms = np.zeros((num_clients))
                for batch_idx, data in enumerate(zip(*clients_dataloader)):
                    for idx in range(num_clients):
                        input, label = data[idx]
                        input, label = input.to(device), label.to(device)
                        former_client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
                        feat1, _, _ = former_client_models[idx](input)
                        now_client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
                        feat2, _, _ = now_client_models[idx](input)
                        bsz, channel, h, w= feat1.shape
                        act1= feat1.reshape((bsz*h*w, bsz))
                        bsz, channel, h, w = feat2.shape
                        act2 = feat2.reshape((bsz * h * w, bsz))
                        f_results = get_cca_similarity(act1.T, act2.T, epsilon=1e-10, verbose=False)
                        norms[idx]+= f_results["cca_coef1"]
                cli_norms = norms.argsort()
                print(norms)
                straggler_list = cli_norms[:num_straggler] #[-num_straggler:]
                client_list = cli_norms[num_straggler:] #[:-num_straggler]
                was_cli[client_list] = 1
                f = open("D:/Dropbox/나메렝/wml/210827/scheduling/svcca_cli5.txt", 'a')
                f.write(np.array2string(norms))
                f.write(np.array2string(straggler_list))
                f.close()

        return client_list, straggler_list



    def transforms_images(self, image_tensor):
        device = image_tensor.device
        _, _, H, W = image_tensor.shape

        cr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((H, W)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        image_tensor = torch.cat([cr_transforms(image).unsqueeze_(0)
                                  for image in image_tensor.cpu()], dim=0)

        return image_tensor.to(device)

    def choose_best_clients(self, epoch, client_list, client_acc, number_of_clients):
        # choose best client who sends feature to straggler (for CT loss)
        if epoch == 0:
            best_client_list = tf.random.shuffle(client_list)[0:number_of_clients]
        else:
            list = []
            for i in client_list:
                list.append(client_acc[i][-1])
            best_client_list = np.argsort(list)[0:number_of_clients]

        return best_client_list


    def straggler_update1(self, straggler_idx, best_idx_list, data):
        # distill from server output
        inputs, labels = data[straggler_idx]
        inputs, labels = inputs.to(device), labels.to(device)

        # # loss_1= criterion_KL(out_pr_stg, label, teacher_pr)
        # # loss_1 = 0.0
        optimizer_c[straggler_idx].zero_grad()
        optimizer_p[straggler_idx].zero_grad()

        feature, out_stg, _ = client_models[straggler_idx](inputs)

        for idx in best_idx_list:
            inputs, labels = data[idx]
            inputs, labels = inputs.to(device), labels.to(device)
            feat, _, _ = client_models[idx](inputs)
            outs, _ = server(feat)
            if idx==best_idx_list[0]:
                server_out=outs
            else:
                server_out+=outs
        server_out = server_out/len(best_idx_list)

        loss = F.kl_div(
                F.log_softmax(out_stg / Temperature, dim=1),
                F.log_softmax(server_out / Temperature, dim=1),
                reduction='sum',
                log_target=True
            )

        # # loss_1 = loss_1 + loss
        loss.backward()
        #
        # if torch.isnan(loss) == True:
        #     print('Warning: loss nan')
        optimizer_c[straggler_idx].step()
        optimizer_p[straggler_idx].step()
        return loss



    def straggler_update2(self, straggler_idx, best_idx_list, data):
        # distill from avg of non-stragglers activation + straggler's classifier layer
        #aa must__ label sync

        inputs, labels = data[straggler_idx]
        inputs, labels = inputs.to(device), labels.to(device)

        # loss_2 = 0.0
        optimizer_c[straggler_idx].zero_grad()
        optimizer_p[straggler_idx].zero_grad()

        feature, _, _ = client_models[straggler_idx](inputs)
        _, out_kl_stg= pr_head(feature)

        for idx in best_idx_list:
            inputs, labels = data[idx]
            inputs, labels = inputs.to(device), labels.to(device)
            best_model.load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
            feat, _, _ = best_model(inputs)
            _, client_act = pr_head(feat)
            if idx==best_idx_list[0]:
                out_pr=client_act
            else:
                out_pr+=client_act
        out_pr=out_pr/len(best_idx_list)

        loss = F.kl_div(
                F.log_softmax(out_kl_stg / Temperature, dim=1),
                F.log_softmax(out_pr / Temperature, dim=1),
                reduction='sum',
                log_target=True
            )
        loss.backward()
        #
        # if torch.isnan(loss) == True:
        #     print('Warning: loss nan')
        optimizer_c[straggler_idx].step()
        optimizer_p[straggler_idx].step()

        return loss



    def straggler_update3(self, straggler_idx, best_idx_list, data):
        # contrastive with non_stragglers activation+straggler's projection layer

        inputs, labels = data[straggler_idx]
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_c[straggler_idx].zero_grad()

        # augmentation own data for positive & negative samples
        feature, _, _ = client_models[straggler_idx](inputs)
        out_ct_stg, _ = pr_head(feature)
        out_ct_stg = F.normalize(out_ct_stg, dim=1)
        out_ct_stg = out_ct_stg.unsqueeze(1)

        feature_aug, _, _ = client_models[straggler_idx](self.transforms_images(inputs))
        out_ct_stg_aug, _ = pr_head(feature_aug)
        out_ct_stg_aug = F.normalize(out_ct_stg_aug, dim=1)
        out_ct_stg_aug = out_ct_stg_aug.unsqueeze(1)

        out = torch.cat([out_ct_stg, out_ct_stg_aug], dim=0)
        label = torch.cat([labels, labels])

        # Positive & negative samples from best clients
        for idx in range(len(best_idx_list)):
            inputs, labels = data[idx]
            inputs, labels = inputs.to(device), labels.to(device)
            client_models[idx].load_state_dict(torch.load('./weight_last_{}.pth'.format(idx)))
            feat, _, _= client_models[idx](inputs)
            client_models[idx].load_state_dict(torch.load('./weight_{}.pth'.format(idx)))
            out_ct, _ = pr_head(feat)
            out_ct = F.normalize(out_ct, dim=1)
            out_ct = out_ct.unsqueeze(1)

            out = torch.cat([out.clone(), out_ct], dim=0)
            label = torch.cat([label.clone(), labels])

        # Supcon loss (SimCLR is in Appendix)
        out2 = F.normalize(out, dim=2)
        loss_3 = criterion_CT(out2, label)
        loss_3.backward()
        #
        # if torch.isnan(loss_3) == True:
        #     print('Warning: loss_3 nan')
        optimizer_c[straggler_idx].step()
        return loss_3


    def straggler_update4(self, straggler_idx, best_idx_list, data):
        # backprop with avg of non-stragglers' gradient

        # for p in client_models[straggler_idx].parameters():
        #     p.grad = torch.zeros_like(p.grad)

        # for param_stg, param in zip(client_models[straggler_idx].parameters(), client_models[best_idx_list[0]].parameters()):
        #     param_stg.grad = param.grad

        for i in best_idx_list:
            if i == best_idx_list[0]:
                for param_stg, param in zip(client_models[straggler_idx].parameters(), client_models[i].parameters()):
                    if param.grad==None and param_stg.grad==None:
                        continue
                    else:
                        param_stg.grad = param.grad/len(best_idx_list)
            else:
                for param_stg, param in zip(client_models[straggler_idx].parameters(), client_models[i].parameters()):
                    if param.grad == None and param_stg.grad == None:
                        continue
                    else:
                        param_stg.grad += param.grad/len(best_idx_list)



    def straggler_update5(self, straggler_idx, best_idx_list, data):
        # local update
        inputs, labels = data[straggler_idx]
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_c[straggler_idx].zero_grad()

        _, out, _ = client_models[straggler_idx](inputs)
        loss_5= criterion_CE(out, labels)
        loss_5.backward()
        optimizer_c[straggler_idx].step()

        return loss_5



    def straggler_update__(self, straggler_idx, best_idx_list, data, features):
        #mse loss
        inputs, labels = data[straggler_idx]
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_c[straggler_idx].zero_grad()
        loss_mse = nn.MSELoss()

        feature, outputs, out_ct = client_models[straggler_idx](inputs)

        loss_m = loss_mse(feature, features.to(device))
        loss_m.backward()

        optimizer_c[straggler_idx].step()



    def client_aggregate(self, client_list):
        # with torch.no_grad():
        #     for param in client.parameters():
        #         param = 0.0
        #     w_avg= client.state_dict()
        #     for key in w_avg.keys():
        #         for client_idx in client_list:
        #             w= client_models[client_idx].state_dict()
        #             w_avg[key] += w[key]
        #         w_avg[key] = torch.div(w_avg[key], len(client_list))
        #
        # for client_idx in client_list:
        #     model_dict = client_models[client_idx].state_dict()
        #     model_dict.update(w_avg)

        with torch.no_grad():
            w1 = client_models[client_list[0]].state_dict()
            for client_idx in client_list:
                exec('weight_{} = client_models[client_idx].state_dict()'.format(client_idx))
            for key in eval('weight_{}'.format(client_list[0])):
                exec('weight_{}[key] = weight_{}[key]/len(client_list)'.format(client_list[0], client_list[0]))
                for client_idx in client_list:
                    exec('weight_{}[key] += weight_{}[key]/len(client_list)'.format(client_list[0], client_idx))

            if agg_to_stg==True:
                for client_idx in range(num_clients):
                    client_models[client_idx].load_state_dict(eval('weight_{}'.format(client_list[0])))
            else:
                for client_idx in client_list:
                    client_models[client_idx].load_state_dict(eval('weight_{}'.format(client_list[0])))





    def local_parallel(self, epoch, num_clients, client_loss, client_acc):
        start = time.time()

        # TODO training
        for batch_idx, data in enumerate(zip(*clients_dataloader)):
            for client_idx in range(num_clients):
                inputs, labels = data[client_idx]
                inputs, labels = inputs.to(device), labels.to(device)
                # client_models[client_idx].train()
                optimizer_c[client_idx].zero_grad()
                optimizer_d[client_idx].zero_grad()
                # feature, outputs, out_ct = client_models[client_idx](inputs)
                feature, _, _ = client_models[client_idx](inputs)
                _, outputs = pr_heads[client_idx](feature)
                loss_ce = criterion_CE(outputs, labels)

                decoder_out= decoders[client_idx](feature)
                loss_bce= criterion_BCE(decoder_out, inputs)
                loss_mse= criterion_MSE(decoder_out, inputs)

                loss_aug=0.0
                if (client_idx in straggler_list) and (1 in straggler_update):
                    loss1=self.straggler_update1(client_idx, self.best_client_list, data)
                    loss_aug+=loss1
                if (client_idx in straggler_list) and (2 in straggler_update):
                    loss2=self.straggler_update2(client_idx, self.best_client_list, data)
                    loss_aug+=loss2
                if (client_idx in straggler_list) and (3 in straggler_update):
                    loss3=self.straggler_update3(client_idx, self.best_client_list, data)
                    loss_aug+=loss3

                if bce==True:
                    loss= 0.6*loss_ce +0.4*loss_bce
                else:
                    loss= loss_ce

                # loss = loss_ce + loss_bce + loss_aug
                loss.backward(retain_graph=True)

                optimizer_c[client_idx].step()
                optimizer_d[client_idx].step()


        end = time.time()



        # TODO test
        with torch.no_grad():
            c_loss, c_acc = torch.zeros((num_clients)).to(device), torch.zeros((num_clients)).to(device)
            for batch_idx, (inputs, labels) in enumerate(testloader, 0):
                inputs, labels = inputs.to(device), labels.to(device)

                for num in range(num_clients):
                    feature, _, out_ct = client_models[num](inputs)
                    _, out = pr_heads[num](feature)
                    loss = criterion_CE(out, labels)
                    _, preds = torch.max(out.data, 1)

                    c_loss[num] += loss.item()
                    c_acc[num] += torch.sum(preds == labels.data)

        print('\nEpoch {}: local parallel      time: {}sec'.format(epoch, round(end - start, 2)))
        for num in range(num_clients):
            c_loss_ = c_loss / len(testloader.dataset)
            client_loss[num].append(c_loss_[num])

            c_acc_ = c_acc.double() / len(testloader.dataset)
            client_acc[num].append(c_acc_[num])
            print('client{}  loss: {}, acc: {}'.format(num, c_loss_[num].item(), c_acc_[num].item()))







    def split_train2(self, epoch, client_list, straggler_list, client_loss, client_acc, final_loss, final_acc):
        # per epoch
        start = time.time()


        # TODO training
        for batch_idx, data in enumerate(zip(*clients_dataloader)):

            if batch_idx==0:
                last_model=client_models
            client_list, straggler_list = self.select_straggler3(num_straggler, last_model, client_models, data)

            last_model = [student_model(Temperature).to(device) for _ in range(num_clients)]
            for i in range(num_clients):
                torch.save(client_models[i].state_dict(), './weight_{}.pth'.format(i))
                last_model[i].load_state_dict(torch.load('./weight_{}.pth'.format(i)))


            for client_idx in client_list:

                inputs, labels = data[client_idx]
                inputs, labels = inputs.to(device), labels.to(device)

                #bb client forward
                optimizer_c[client_idx].zero_grad()
                exec('feature_{}, out, out_ct= client_models[client_idx](inputs)'.format(client_idx))
                exec('self.inter_to_server_{} = feature_{}.detach().requires_grad_()'.format(client_idx, client_idx))
                #bb server forward
                optimizer_s.zero_grad()
                out, _= server(eval('self.inter_to_server_{}'.format(client_idx)))
                exec('loss_{}=criterion_CE(out, labels)'.format(client_idx))
                exec('loss_{}.backward(retain_graph=True)'.format(client_idx))
                #bb client backward
                grad_to_client = eval('self.inter_to_server_{}.grad.clone()'.format(client_idx)) ######
                exec('feature_{}.backward(grad_to_client)'.format(client_idx))
                optimizer_c[client_idx].step()
            #bb server bakward
            optimizer_s.zero_grad()
            for i in client_list:
                if i==client_list[0]:
                    loss= eval('loss_{}'.format(i))
                else:
                    loss= loss+eval('loss_{}/len(client_list)'.format(i))

            if server_update == True:
                for stg_idx in straggler_list:
                    if was_cli[stg_idx] == 1:
                        out, _ = server(eval('self.inter_to_server_{}'.format(stg_idx)))
                        exec('loss_{}=criterion_CE(out, labels)'.format(stg_idx))
                        loss= loss+eval('loss_{}/len(client_list)'.format(stg_idx))
            loss.backward()
            optimizer_s.step()

            if per_batch==True:
                if randomly==False:
                    client_list, straggler_list = self.select_straggler3(num_straggler, last_model, client_models, data)




        # self.best_client_list = self.choose_best_clients(epoch, client_list, client_acc, num_best_clients)

        # #TODO straggler update
        for local_epoch in range(straggler_epochs):
            for batch_idx, data in enumerate(zip(*clients_dataloader)):
                for straggler_idx in straggler_list:
                    best_client_list = self.choose_best_clients(epoch, client_list, client_acc, num_best_clients)

                    if 1 in straggler_update:
                        loss__1= self.straggler_update1(straggler_idx, best_client_list, data)
                        loss= loss+loss__1
                    if 2 in straggler_update:
                        loss__2= self.straggler_update2(straggler_idx, best_client_list, data)
                        loss = loss + loss__2
                    if 3 in straggler_update:
                        for e in range(num_best_clients):
                            loss__3= self.straggler_update3(straggler_idx, best_client_list, data)
                            loss = loss + loss__3
                    if 4 in straggler_update:
                        self.straggler_update4(straggler_idx, best_client_list, data)

                    if 5 in straggler_update:
                        loss__5= self.straggler_update5(straggler_idx, best_client_list, data)
                        loss = loss + loss__5
                    else:
                        continue
                    # loss.backward()
                    # optimizer_c[straggler_idx].step()


        end = time.time()

        # TODO test
        with torch.no_grad():
            c_loss, c_acc = torch.zeros((num_clients)).to(device), torch.zeros((num_clients)).to(device)

            for batch_idx, (inputs, labels) in enumerate(testloader, 0):
                inputs, labels = inputs.to(device), labels.to(device)

                for num in range(num_clients):
                    feature, out, out_kd = client_models[num](inputs)
                    feature_to_server = feature.detach()
                    out, _ = server(feature_to_server)
                    loss = criterion_CE(out, labels)
                    _, preds = torch.max(out.data, 1)

                    c_loss[num] += loss.item()
                    # c_acc[num] += preds.eq(labels.view_as(preds)).sum().item()
                    c_acc[num] += torch.sum(preds == labels.data)

            print('\nEpoch {}: split training      time: {}sec'.format(epoch, round(end - start, 2)))
            print(client_list, "    ", straggler_list)
            for num in range(num_clients):
                c_loss_ = c_loss / len(testloader.dataset)
                client_loss[num].append(c_loss_[num])
                c_acc_ = c_acc.double() / len(testloader.dataset)
                client_acc[num].append(c_acc_[num])
                print('client{} loss: {}, acc: {}'.format(num, c_loss_[num].item(), c_acc_[num].item()))

        # TODO client weight averaging
        if federated == True:
            if entire_aggregate == True:
                self.client_aggregate(range(num_clients))  # client_list+straggler_list
            elif entire_aggregate == False:
                self.client_aggregate(client_list)

            with torch.no_grad():
                c_loss, c_acc = 0.0, 0.0
                loss_test1, acc_test1, loss_test2, acc_test2 = 0.0, 0.0, 0.0, 0.0


                for batch_i, (data1, data2) in enumerate(zip(testloader, testloader)):
                    with torch.no_grad():
                        input1, label1 = data1
                        input2, label2 = data2
                        input1, label1 = input1.to(device), label1.to(device)
                        input2, label2 = input2.to(device), label2.to(device)
                        feature, out, out_kd = client_models[0](input1)
                        out1, _ = server(feature)
                        loss1 = criterion_CE(out1, label1)
                        _, preds_s1 = torch.max(out1.data, 1)

                        loss_test1 += loss1.item()
                        acc_test1 += torch.sum(preds_s1 == label1.data)

                        feature, out, out_kd = client_models[1](input2)
                        out2, _ = server(feature)
                        loss2 = criterion_CE(out2, label2)
                        _, preds_s2 = torch.max(out2.data, 1)

                        loss_test2 += loss2.item()
                        acc_test2 += torch.sum(preds_s2 == label2.data)

                _loss_test1 = loss_test1 / len(testloader.dataset)
                _acc_test1 = acc_test1 / len(testloader.dataset)
                _loss_test2 = loss_test2 / len(testloader.dataset)
                _acc_test2 = acc_test2 / len(testloader.dataset)
                final_acc.append(_acc_test2)
                final_loss.append(_loss_test2)
            print('Epoch {}\nrenset1:    loss: {}, acc: {}\nresnet2:    loss: {}, acc: {}'.format(epoch, _loss_test1,
                                                                                                  _acc_test1,
                                                                                                  _loss_test2,
                                                                                                  _acc_test2))

        return server, client_models





    def split_train3(self, epoch, client_list, straggler_list, client_loss, client_acc, final_loss, final_acc):
        # per batch
        start = time.time()
        # giveuppp= [[] for i in range(num_clients)] #####################################################################
        # for i in range(num_clients): ###################################################################################
        #     giveuppp[i]= np.random.choice(100,int(giveups_co45[epoch]),replace=False)###################################

        for batch_idx, data in enumerate(zip(*clients_dataloader)):
            loss_avg=0.0
            c=0
            for client_idx in client_list:

                inputs, labels = data[client_idx]
                inputs, labels = inputs.to(device), labels.to(device)

                #bb client forward
                optimizer_c[client_idx].zero_grad()
                feature, out, out_ct= client_models[client_idx](inputs)
                inter_to_server = feature.detach().requires_grad_()

                # Sensoring
                transmit=True
                if epoch==0 or select==None:
                    exec('self.inter_to_server_{}_{} = inter_to_server'.format(client_idx, batch_idx))
                    transmit=True

                elif select=='random':
                    diff=random.choice((0,1))
                    if diff==0:
                        transmit=False

                elif select=='l2norm':
                    diff= torch.norm((inter_to_server-eval('self.inter_to_server_{}_{}'.format(client_idx, batch_idx))), 'fro')
                    if diff<threshold:
                        transmit=False

                elif select=='cosine':
                    act1, _= pr_head(inter_to_server)
                    act2, _= pr_head(eval('self.inter_to_server_{}_{}'.format(client_idx, batch_idx)))
                    cos= torch.nn.CosineEmbeddingLoss()
                    diff= cos(act1, act2, labels)
                    if diff<threshold:
                        transmit=False
                # if batch_idx in giveuppp[client_idx]: ##################################################################
                #     transmit=False######################################################################################

                if transmit==False:
                    giveups[client_idx, epoch]+=1





                # communication fail
                fail= random.choices([True, False], weights=(fail_weight, 1-fail_weight), k=1)[0]



                # bb server forward
                if transmit==True and fail==False:
                    exec('self.inter_to_server_{}_{}= inter_to_server'.format(client_idx, batch_idx))
                elif transmit==True and fail==True:
                    exec('self.inter_to_server_{}_{}= mixup(self.inter_to_server_{}_{}, labels)'.format(client_idx, batch_idx, client_idx, batch_idx))

                optimizer_s.zero_grad()
                # if transmit==True:  #####################이거빼고 else 밑까지 shift+tab
                out, _= server(eval('self.inter_to_server_{}_{}'.format(client_idx, batch_idx)))
                loss=criterion_CE(out, labels)
                loss.backward(retain_graph=True)


                if transmit==True and fail==False:
                    loss_avg= loss_avg+loss
                    c+=1

                #bb client backward
                if transmit==True and fail==False:
                    grad_to_client = inter_to_server.grad.clone() ######
                    feature.backward(grad_to_client)
                    optimizer_c[client_idx].step()
                elif transmit == True and fail == True:
                    best_client_list = self.choose_best_clients(epoch, client_list, client_acc, num_best_clients)
                    if 1 in straggler_update:
                        self.straggler_update1(client_idx, best_client_list, data)
                    if 2 in straggler_update:
                        self.straggler_update2(client_idx, best_client_list, data)
                    if 3 in straggler_update:
                        self.straggler_update3(client_idx, best_client_list, data)
                    if 4 in straggler_update:
                        self.straggler_update4(client_idx, best_client_list, data)
                    if 5 in straggler_update:
                        self.straggler_update5(client_idx, best_client_list, data)
                    else:
                        continue

            #bb server backward
            if c!=0:
                loss_avg= loss_avg/c
                loss_avg.backward()
                optimizer_s.step()




        end = time.time()

        # TODO test
        with torch.no_grad():
            c_loss, c_acc = torch.zeros((num_clients)).to(device), torch.zeros((num_clients)).to(device)

            for batch_idx, (inputs, labels) in enumerate(testloader, 0):
                inputs, labels = inputs.to(device), labels.to(device)

                for num in range(num_clients):
                    feature, out, out_kd = client_models[num](inputs)
                    feature_to_server = feature.detach()
                    out, _ = server(feature_to_server)
                    loss = criterion_CE(out, labels)
                    _, preds = torch.max(out.data, 1)

                    c_loss[num] += loss.item()
                    # c_acc[num] += preds.eq(labels.view_as(preds)).sum().item()
                    c_acc[num] += torch.sum(preds == labels.data)

            print('\nEpoch {}: split training      time: {}sec'.format(epoch, round(end - start, 2)))
            print(giveups[:, epoch])
            for num in range(num_clients):
                c_loss_ = c_loss / len(testloader.dataset)
                client_loss[num].append(c_loss_[num].cpu())
                c_acc_ = c_acc.double() / len(testloader.dataset)
                client_acc[num].append(c_acc_[num].cpu())
                print('client{} loss: {}, acc: {}'.format(num, c_loss_[num].item(), c_acc_[num].item()))

        # TODO client weight averaging
        if federated == True:
            if entire_aggregate == True:
                self.client_aggregate(range(num_clients))  # client_list+straggler_list
            elif entire_aggregate == False:
                self.client_aggregate(client_list)

            with torch.no_grad():
                c_loss, c_acc = 0.0, 0.0
                loss_test1, acc_test1, loss_test2, acc_test2 = 0.0, 0.0, 0.0, 0.0


                for batch_i, (data1, data2) in enumerate(zip(testloader, testloader)):
                    with torch.no_grad():
                        input1, label1 = data1
                        input2, label2 = data2
                        input1, label1 = input1.to(device), label1.to(device)
                        input2, label2 = input2.to(device), label2.to(device)
                        feature, out, out_kd = client_models[0](input1)
                        out1, _ = server(feature)
                        loss1 = criterion_CE(out1, label1)
                        _, preds_s1 = torch.max(out1.data, 1)

                        loss_test1 += loss1.item()
                        acc_test1 += torch.sum(preds_s1 == label1.data)

                        feature, out, out_kd = client_models[1](input2)
                        out2, _ = server(feature)
                        loss2 = criterion_CE(out2, label2)
                        _, preds_s2 = torch.max(out2.data, 1)

                        loss_test2 += loss2.item()
                        acc_test2 += torch.sum(preds_s2 == label2.data)

                _loss_test1 = loss_test1 / len(testloader.dataset)
                _acc_test1 = acc_test1 / len(testloader.dataset)
                _loss_test2 = loss_test2 / len(testloader.dataset)
                _acc_test2 = acc_test2 / len(testloader.dataset)
                final_acc.append(_acc_test2)
                final_loss.append(_loss_test2)
            print('Epoch {}\nrenset1:    loss: {}, acc: {}\nresnet2:    loss: {}, acc: {}'.format(epoch, _loss_test1,
                                                                                                  _acc_test1,
                                                                                                  _loss_test2,
                                                                                                  _acc_test2))

        return server, client_models















# todo main_________________________________________________________________________________________________________
#per Batch
# for i in range(num_clients):
#     client_models[i].load_state_dict(torch.load('./initial_cli_{}.pth'.format(i)))
# server.load_state_dict(torch.load('./initial_ser.pth'))

client_loss = [[] for _ in range(num_clients)]
client_acc = [[] for _ in range(num_clients)]
final_loss = []
final_acc = []
federated = False
entire_aggregate = True
agg_to_stg=True

total_epoch = 150
# total_period = 6
# client_epoch = 5
# num_straggler = 5
num_best_clients=3


distill = Distill(total_epoch)

#sensoring
select= None
threshold=0.0045
giveups=np.zeros((num_clients, total_epoch))

#comm fail
fail_weight=0
straggler_update=[]
straggler_epochs=1


was_cli = np.zeros((num_clients))

client_list = np.arange(num_clients)
straggler_list = []

for epoch in range(total_epoch):
    _, client_models = distill.split_train3(epoch, client_list, straggler_list, client_loss, client_acc, final_loss, final_acc)

    np.save('D:/Dropbox/나메렝/wml/210925/0928/layer3.npy', client_acc)
    # np.save('D:/Dropbox/나메렝/wml/210925/giveups_co_45_2.npy', giveups)







'''
# todo main_________________________________________________________________________________________________________
#per Epoch

for i in range(num_clients):
    client_models[i].load_state_dict(torch.load('./initial_cli_{}.pth'.format(i)))
server.load_state_dict(torch.load('./initial_ser.pth'))

client_loss = [[] for _ in range(num_clients)]
client_acc = [[] for _ in range(num_clients)]
final_loss = []
final_acc = []
federated = False
entire_aggregate = True
agg_to_stg=True

total_epoch = 100
total_period = 6
client_epoch = 5
num_straggler = 5
num_best_clients=3
bce = True  #no
per_batch = False
randomly = True

distill = Distill(total_epoch)
server_update = True
straggler_update=[5]
straggler_epochs=1


was_cli = np.zeros((num_clients))

client_list = np.arange(num_clients)
straggler_list = []

for epoch in range(total_epoch):
    if epoch % total_period < 0:  # >= total_period - client_epoch and epoch != total_epoch-1:
        distill.local_parallel(epoch, num_clients, client_loss, client_acc)

    else:

        if per_batch == False:
            for i in client_list:
                torch.save(client_models[i].state_dict(), './weight_{}.pth'.format(i))

            if randomly == True:
                client_list, straggler_list = distill.select_straggler(num_straggler)

        _, client_models = distill.split_train2(epoch, client_list, straggler_list, client_loss, client_acc, final_loss, final_acc)

        for i in range(num_clients):
            torch.save(client_models[i].state_dict(), './weight_last_{}.pth'.format(i))

        if per_batch == False:
            if randomly == False:
                client_list, straggler_list = distill.select_straggler3(num_straggler, client_models, client_models, epoch)

    np.save('D:/Dropbox/나메렝/wml/210827/scheduling/norm_perclient.npy', client_acc)
    # np.save('D:/Dropbox/나메렝/wml/210827/scheduling/3_final.npy', final_acc)

'''










# TODO Appendix
'''
x_len= np.arange(epoch)
for num in np.arange(num_clients-1):
    plt.plot(x_len, client_acc[num], 'blue') 

plt.plot(x_len, client_acc[-1], 'blue', label='clients')
plt.plot(x_len, server_acc, color='red', linewidth=4, label='server')
plt.legend()
plt.grid()
plt.show()



out_ct_2= torch.cat([out_ct, out_ct_best],0)
out_ct_3= torch.cat([out_ct_aug, out_ct_best],0)
out_simclr= torch.cat([out_ct_2, out_ct_3],1)    # loss=criterion_CTT(out, label), 크기(bsz,2,pr head) 로 맞춰야해서 aug 는 cat(dim=1)
'''
