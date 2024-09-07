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
from train import DistillationLoss, SupConLoss, one_hot, mixup
from dataset import batchsize, SplitData, testloader, num_datasplits, new_dataload, NoniidSplitData
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import torch.nn.functional as F
import collections
import random
import time
import math
import copy
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO Parameters
lr = 0.01
lr_c = 0.01
lr_decay = 0.95
num_clients = 8
alpha = 1
distribute_alpha=0.5
Temperature = 7




# TODO model
server = teacher_model().to(device)
# client = student_model(Temperature).to(device)

# server_models = [teacher_model().to(device) for _ in range(num_clients)]
client_models = [student_model(Temperature).to(device) for _ in range(num_clients)]
pr_heads = [projection_head().to(device) for _ in range(num_clients)]
decoders= [Decoder(feature_size[0], feature_size[1]).to(device) for _ in range(num_clients)]

optimizer_s = optim.SGD(server.parameters(), lr=lr, momentum=0.9)
# scheduler_s = optim.lr_scheduler.LambdaLR(optimizer=optimizer_s, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_epoch=-1, verbose=False)

# optimizer_ss = [optim.SGD(model.parameters(), lr=lr_c, momentum=0.9) for model in server_models]
# scheduler_ss = [optim.lr_scheduler.LambdaLR(optimizer=optimiz, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_epoch=-1, verbose=False) for optimiz in optimizer_ss]

# optimizer_c = [optim.SGD(model.parameters(), lr=lr_c, momentum=0.9) for model in client_models]
optimizer_c = [optim.Adam(model.parameters(), lr=lr_c) for model in client_models]
# scheduler_c = [optim.lr_scheduler.LambdaLR(optimizer=optimiz, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_epoch=-1, verbose=False) for optimiz in optimizer_c]
optimizer_p = [optim.Adam(model.parameters(), lr=lr_c) for model in pr_heads]
optimizer_d = [optim.Adam(model.parameters(), lr=0.001) for model in decoders]


# TODO critierion
criterion_KL = DistillationLoss(distillation_type='soft', alpha=alpha, tau=Temperature)
criterion_CE = nn.CrossEntropyLoss()
criterion_CT = SupConLoss(contrast_mode='all', base_temperature=0.07)
criterion_BCE= nn.BCELoss()
criterion_MSE= nn.MSELoss()
# criterion_CT= supervised_nt_xent_loss()   #label, feature 순 .cpu().detach().numpy()


# TODO dataset per clients

new_dataload(True)
# clients_dataset1 = [SplitData(i, 0) for i in range(num_clients)]

# clients_dataset1 = [SplitData(i, 0) for i in range(int(num_clients/2))]
# clients_dataset2= [SplitData(i,1) for i in range(int(num_clients/2))]
# clients_dataset= clients_dataset1+clients_dataset2
# clients_dataset = clients_dataset1
# clients_dataloader = [DataLoader(dataset, batch_size=batchsize, shuffle=False) for dataset in clients_dataset]

clients_dataset= NoniidSplitData(int(50000/num_clients), num_clients, distribute_alpha)
clients_dataloader = [DataLoader(dataset, batch_size=batchsize, shuffle=True) for dataset in clients_dataset]





class Distill:
    def __init__(self, epoch):
        self.epoch = epoch

    def select_straggler(self, num_straggler):
        self.num_straggler = num_straggler
        straggler_list = random.sample(list(np.arange(num_clients)), self.num_straggler)
        client_list = [x for x in list(np.arange(num_clients)) if (x not in straggler_list)]

        return client_list, straggler_list

    def select_straggler2(self, num_straggler, former_client_models, now_client_models):
        norms= [[] for i in range(num_clients)]
        for client_idx in range(num_clients):
            for param1, param2 in zip(former_client_models[client_idx].parameters(), now_client_models[client_idx].parameters()):
                norms[client_idx].append(torch.dist(param1, param2).cpu().detach().numpy())
        norm= np.sum(norms,1)
        max_idx= np.argsort(norm)
        straggler_list= max_idx[:num_straggler]
        client_list= max_idx[num_straggler:]
        print(norm)

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
        # optimizer_c[straggler_idx].zero_grad()
        # optimizer_p[straggler_idx].zero_grad()

        feature, _, _ = client_models[straggler_idx](inputs)
        _, out_stg = pr_heads[straggler_idx](feature)

        with torch.no_grad():
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
        # loss.backward()
        #
        # if torch.isnan(loss) == True:
        #     print('Warning: loss nan')
        # optimizer_c[straggler_idx].step()
        # optimizer_p[straggler_idx].step()
        return loss



    def straggler_update2(self, straggler_idx, best_idx_list, data):
        # distill from non-stragglers activation+straggler's classifier layer
        inputs, labels = data[straggler_idx]
        inputs, labels = inputs.to(device), labels.to(device)

        # loss_2 = 0.0
        # optimizer_c[straggler_idx].zero_grad()
        # optimizer_p[straggler_idx].zero_grad()

        feature, _, _ = client_models[straggler_idx](inputs)
        _, out_kl_stg = pr_heads[straggler_idx](feature)

        with torch.no_grad():
            for idx in best_idx_list:
                inputs, labels = data[idx]
                inputs, labels = inputs.to(device), labels.to(device)
                feat, _, _ = self.last_client(inputs)
                _, client_act = pr_heads[straggler_idx](feat)
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
        # loss.backward()
        #
        # if torch.isnan(loss) == True:
        #     print('Warning: loss nan')
        # optimizer_c[straggler_idx].step()
        # optimizer_p[straggler_idx].step()

        return loss



    def straggler_update3(self, straggler_idx, best_idx_list, data):
        # contrastive with non_stragglers activation+straggler's projection layer
        inputs, labels = data[straggler_idx]
        inputs, labels = inputs.to(device), labels.to(device)
        # optimizer_c[straggler_idx].zero_grad()

        # augmentation own data for positive & negative samples
        feature, _, _ = client_models[straggler_idx](inputs)
        out_ct_stg, _ = pr_heads[straggler_idx](feature)
        out_ct_stg = F.normalize(out_ct_stg, dim=1)
        out_ct_stg = out_ct_stg.unsqueeze(1)

        feature_aug, _, _ = client_models[straggler_idx](self.transforms_images(inputs))
        out_ct_stg_aug, _ = pr_heads[straggler_idx](feature_aug)
        out_ct_stg_aug = F.normalize(out_ct_stg_aug, dim=1)
        out_ct_stg_aug = out_ct_stg_aug.unsqueeze(1)

        out = torch.cat([out_ct_stg, out_ct_stg_aug], dim=0)
        label = torch.cat([labels, labels])

        # Positive & negative samples from best clients
        with torch.no_grad():
            for idx in range(len(best_idx_list)):
                inputs, labels = data[idx]
                inputs, labels = inputs.to(device), labels.to(device)
                feat, _, _= self.last_client(inputs)
                out_ct, _ = pr_heads[straggler_idx](feat)
                out_ct = F.normalize(out_ct, dim=1)
                out_ct = out_ct.unsqueeze(1)

                out = torch.cat([out.clone(), out_ct], dim=0)
                label = torch.cat([label.clone(), labels])

        # Supcon loss (SimCLR is in Appendix)
        out2 = F.normalize(out, dim=2)
        loss_3 = criterion_CT(out2, label)
        # loss_3.backward()
        #
        # if torch.isnan(loss_3) == True:
        #     print('Warning: loss_3 nan')
        # optimizer_c[straggler_idx].step()
        return loss_3


    def straggler_update4(self, straggler_idx, best_idx_list, data):
        # backprop with avg of non-stragglers' gradient
        # for p in client_models[straggler_idx].parameters():
        #     p.grad = torch.zeros_like(p.grad)
        for param_stg, param in zip(client_models[straggler_idx].parameters(), client_models[best_idx_list[0]].parameters()):
            param_stg.grad = param.grad

        # for i in best_idx_list:
        #     if i == best_idx_list[0]:
        #         for param_stg, param in zip(client_models[straggler_idx].parameters(), client_models[i].parameters()):
        #             param_stg.grad = param.grad/len(best_idx_list)
        #     else:
        #         for param_stg, param in zip(client_models[straggler_idx].parameters(), client_models[i].parameters()):
        #             param_stg.grad += param.grad/len(best_idx_list)

        optimizer_c[straggler_idx].step()



    def straggler_update__(self, straggler_idx, best_idx_list, data, features):
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

                loss= 0.6*loss_ce +0.4*loss_bce
                # loss= loss_ce
                # loss = loss_ce + loss_bce + loss_aug
                loss.backward(retain_graph=True)

                optimizer_c[client_idx].step()
                optimizer_d[client_idx].step()

                # if (client_idx in straggler_list) and (1 in straggler_update):
                #     self.straggler_update1(client_idx, self.best_client_list, data)
                # if (client_idx in straggler_list) and (2 in straggler_update):
                #     self.straggler_update2(client_idx, self.best_client_list, data)
                # if (client_idx in straggler_list) and (3 in straggler_update):
                #     self.straggler_update3(client_idx, self.best_client_list, data)

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





    def split_train(self, epoch, client_list, straggler_list, straggler_update, client_loss, client_acc, final_loss, final_acc, method='avgpool', federated=False):
        start = time.time()
        self.last_client= client_models[0]


        # TODO training
        for batch_idx, data in enumerate(zip(*clients_dataloader)):

            for i in client_list:
                optimizer_c[i].zero_grad()   ##########

            for client_idx in client_list:

                inputs, labels = data[client_idx]
                inputs, labels = inputs.to(device), labels.to(device)
                exec('self.labels_{} = labels.clone()'.format(client_idx))
                # bb client forward
                exec('feature_{}, out, out_ct= client_models[client_idx](inputs)'.format(client_idx))
                exec('self.inter_to_server_{} = feature_{}.detach().requires_grad_()'.format(client_idx, client_idx))




                # bb server forward

                if method_batch_control == True:
                    for e in range(server_epoch):
                        output, teacher_pr = server(eval('self.inter_to_server_{}'.format(client_idx)))
                        loss = criterion_CE(output, labels)
                        # bb server backward
                        loss.backward()
                        optimizer_s.step()
                    # bb client backward
                    exec('grad_to_client_{} = self.inter_to_server_{}.grad.clone()'.format(client_idx, client_idx))
                    exec('feature_{}.backward(grad_to_client_{})'.format(client_idx, client_idx))
                    optimizer_c[client_idx].step()

                elif method_batch_control == False:

                    if client_idx == client_list[0]:
                        intermediate_to_server = eval('self.inter_to_server_{}.clone()'.format(client_idx))
                        label = labels
                    else:
                        if method == 'avgpool':
                            intermediate_to_server += eval('self.inter_to_server_{}.clone()'.format(client_idx))
                            label += labels
                        elif method == 'concat':
                            intermediate_to_server = torch.cat([intermediate_to_server.clone(), eval('self.inter_to_server_{}.clone()'.format(client_idx))], dim=0)
                            label = torch.cat([label.clone(), eval('self.labels_{}'.format(client_idx))])

                    if client_idx == client_list[-1]:
                        if method == 'avgpool':
                            output, _ = server(intermediate_to_server / len(client_list))
                            loss = criterion_CE(output, (label / len(client_list)).long())

                        elif method == 'concat':

                            if server_update1 == True and len(client_list)<num_clients and epoch>1:
                                # augmentation: add gaussian noise
                                for stg_idx in straggler_list:
                                    if was_cli[stg_idx] == 1:
                                        var= torch.var(eval('self.inter_to_server_{}'.format(stg_idx)))
                                        noise= torch.normal(mean=0, std= math.sqrt(var), size= eval('self.inter_to_server_{}.shape'.format(stg_idx)))
                                        exec('inter_to_server_{} = self.inter_to_server_{} + noise.to(device)'.format(stg_idx, stg_idx))
                                        intermediate_to_server = torch.cat([intermediate_to_server.clone(), eval('inter_to_server_{}'.format(stg_idx))], dim=0)
                                        label = torch.cat([label.clone(), eval('self.labels_{}'.format(stg_idx))])

                            if server_update2 == True and len(client_list)<num_clients and epoch>1:
                                # augmentation : manifold mixup classwise
                                for stg_idx in straggler_list:
                                    if was_cli[stg_idx]==1:
                                        exec('inter_to_server_{} = mixup(self.inter_to_server_{}, self.labels_{})'.format(stg_idx, stg_idx, stg_idx))
                                        intermediate_to_server = torch.cat([intermediate_to_server.clone(), eval('inter_to_server_{}'.format(stg_idx))], dim=0)
                                        label = torch.cat([label.clone(), eval('self.labels_{}'.format(stg_idx))])


                        for e in range(server_epoch):
                            optimizer_s.zero_grad()
                            teacher_out, teacher_pr = server(intermediate_to_server)
                            loss = criterion_CE(teacher_out, label.long())

                            # bb server backward
                            loss.backward()
                            optimizer_s.step()


                        # bb client backward
                        for i in client_list:
                            exec('grad_to_client_{} = self.inter_to_server_{}.grad.clone()'.format(i, i))
                            exec('feature_{}.backward(grad_to_client_{})'.format(i, i))
                            optimizer_c[i].step()
                        # for i in straggler_list : #aa augmentation 할거면 여기에 모든 client에 backward 는 다 되게 해야됌
                        #     if was_cli[i]==1 and epoch>1:
                        #         exec('grad_to_client_{} = self.inter_to_server_{}.grad.clone()'.format(i, i))
                        #         feature, _, _ = client_models[i](inputs)
                        #         exec('feature.backward(grad_to_client_{})'.format(i, i))
                        #         optimizer_c[i].step()


        self.best_client_list = self.choose_best_clients(epoch, client_list, client_acc, num_best_clients)

        # #TODO straggler update
        # for local_epoch in range(straggler_epochs):
        #     for batch_idx, data in enumerate(zip(*clients_dataloader)):
        #         for straggler_idx in straggler_list:
        #             best_client_list = self.choose_best_clients(epoch, client_list, client_acc, num_best_clients)
        #
        #             if 1 in straggler_update:
        #                 self.straggler_update1(straggler_idx, self.teacher_out.detach(), data)
        #             if 2 in straggler_update:
        #                 self.straggler_update2(straggler_idx, best_client_list, data)
        #             if 3 in straggler_update:
        #                 for e in range(num_best_clients):
        #                     self.straggler_update3(straggler_idx, best_client_list, data)
        #             if 4 in straggler_update:
        #                 self.straggler_update4(straggler_idx, best_client_list, data)
        #
        #             else:
        #                 continue


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
            if entire_aggregate==True:
                self.client_aggregate(range(num_clients))  # client_list+straggler_list
            elif entire_aggregate ==False:
                self.client_aggregate(client_list)

            with torch.no_grad():
                c_loss, c_acc = 0.0, 0.0
                loss_test1, acc_test1, loss_test2, acc_test2 = 0.0, 0.0,0.0,0.0

                # for batch_idx, (inputs, labels) in enumerate(testloader, 0):
                #     inputs, labels = inputs.to(device), labels.to(device)
                #
                #     feature, out, out_kd = client_models[0](inputs)
                #     feature_to_server = feature.detach()
                #     out, _ = server(feature_to_server)
                #     loss = criterion_CE(out, labels)
                #     _, preds = torch.max(out.data, 1)
                #
                #     c_loss += loss.item()
                #     # c_acc[num] += preds.eq(labels.view_as(preds)).sum().item()
                #     c_acc += torch.sum(preds == labels.data)
                #
                # c_loss_ = c_loss / len(testloader.dataset)
                # final_loss.append(c_loss_)
                # c_acc_ = c_acc.double() / len(testloader.dataset)
                # final_acc.append(c_acc_)
                # print('final loss: {}, acc: {}'.format(c_loss_, c_acc_))

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
            print('Epoch {}\nrenset1:    loss: {}, acc: {}\nrenset2:    loss: {}, acc: {}'.format(epoch, _loss_test1, _acc_test1, _loss_test2, _acc_test2))

            # if epoch==total_epoch-1:
                # torch.save(client_models[0], './client_model_proposed_concat_bce.pth')
                # torch.save(server, './server_model_proposed_concat_bce.pth')

        return server, client_models












#TODO main_________________________________________________________________________________________________________

print('Datasets: {} samples per client (Each client has {} samples per class)'.format((50000 / num_datasplits),(5000 / num_datasplits)))
client_loss = [[] for _ in range(num_clients)]
client_acc = [[] for _ in range(num_clients)]
final_loss = []
final_acc = []
total_epoch = 300
total_period = 5
server_epoch = 1 #####
client_epoch = 4
straggler_epochs = 1
num_straggler = 4
num_best_clients = 4


#split_train options
method= 'concat'
method_batch_control = False
federated = True
server_update1 = False
server_update2 = False
entire_aggregate = True
agg_to_stg = True
straggler_update=[]


distill = Distill(total_epoch)
fail_count = 0  #5번에 4번 통신 안되게 (처음에 되고)
was_cli=np.zeros((num_clients))

for epoch in range(total_epoch):
    former= client_models.copy()
    if epoch % total_period >= total_period - client_epoch and epoch != total_epoch-1:
        distill.local_parallel(epoch, num_clients, client_loss, client_acc)

    else:
        # fail_count += 1  #1부터 시작
        # if fail_count % 10 == 1 or epoch==49:  #1이여야 처음에 되는거임 %5==1   #10
        #     client_list = np.arange(num_clients)
        #     straggler_list = []
        #
        # else:  #communication fail
        #     client_list = np.arange(num_clients)[:-num_straggler]
        #     straggler_list = np.arange(num_clients)[-num_straggler:]

        client_list = np.arange(num_clients)
        straggler_list = []

        # client_list, straggler_list= distill.select_straggler(num_straggler)


        distill.split_train(epoch, client_list, straggler_list, straggler_update, client_loss, client_acc, final_loss, final_acc, method, federated)
        for i in client_list:
            was_cli[i]=1
    # np.save('D:/Dropbox/나메렝/wml/210820/noniid_serverepoch1_perclient.npy', client_acc)
    # np.save('D:/Dropbox/나메렝/wml/210820/noniid_serverepoch1_final.npy', final_acc)

np.save('D:/Dropbox/나메렝/wml/210820/noniid_serverepoch1_perclient.npy', client_acc)
np.save('D:/Dropbox/나메렝/wml/210820/noniid_serverepoch1_final.npy', final_acc)












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
