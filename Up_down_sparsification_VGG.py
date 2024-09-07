import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torchvision
from torchsummary import summary
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.nn import functional as F
import time
import datetime
import copy
import math
import os
import gc
from losses import DistillationLoss
from model_vgg import VGGNet, VGGNet_client, VGGNet_server, VGGNet_tinyserver, VGGNet_tinyserver_student
from dataset import batchsize, SplitData, num_datasplits, new_dataload, NoniidSplitData
import torch.nn.utils.prune as prune
import collections

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


'''parameters'''
learning_rate = 0.1
num_epoch = 150
num_classes = 10
num_clients = 4
criterion = nn.CrossEntropyLoss()
tau = 2
alpha = 0.4
tiny_epoch = 5
threshold = 0.007  # loss1: 0.007 #loss1_kl: 0.0001
epsilon = 0.2
noniid_distribution = 1
warmup_epoch=1
data_distrib = 'noniid'
mode='prunetoteacher_loss1'

# prune to teacher
remain_percent_prune = 18.476  # %
# # prune to student
# remain_percent_prune = 0.44159




'''Train dataset'''
new_dataload(True)
if data_distrib == 'iid':
    clients_dataset1 = [SplitData(i, 0) for i in range(num_datasplits)]
    clients_dataset2 = [SplitData(i, 1) for i in range(num_datasplits)]
    clients_dataset = clients_dataset1 + clients_dataset2
elif data_distrib == 'noniid':
    trainset = SplitData(0, 0)  # num_datasplit= 원래나누려던 숫자/num_clients
    clients_dataset = NoniidSplitData(trainset, int(50000 / (num_datasplits * num_clients)), noniid_distribution, num_clients, num_classes)


for i in range(num_clients):
    y_train = [y for _, y in clients_dataset[i]]
    counter_train = collections.Counter(y_train)
    print(counter_train)


trainloader = [DataLoader(dataset, batch_size=batchsize, shuffle=True) for dataset in clients_dataset]



'''Test dataset'''
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
set_seed(0)
testloader = DataLoader(testset, num_workers=0, shuffle=True, batch_size=batchsize, worker_init_fn=np.random.seed(0))



'''models'''
clients = [VGGNet_client().to(device) for _ in range(num_clients)]
server = VGGNet_server().to(device)
tiny_server = VGGNet_tinyserver().to(device)



opt_c = [optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3) for model in clients]
sch_c = [optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiz, T_max=300) for optimiz in opt_c]
opt_s = optim.SGD(server.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
sch_s = optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=200)
opt_t = optim.SGD(tiny_server.parameters(), lr=learning_rate, momentum=0.9)
sch_t = optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=300)



'''saves'''
batch_num = len(trainloader[0])
batch_num_test = len(testloader)
print('Batch number of each client: ', batch_num)
acc = []
tiny_acc = []
uplinks= np.zeros((num_clients, batch_num*num_epoch))
train_losses_ser = []
test_losses_ser = []
test_losses_tiny = []
tiny_kl_loss_train = []
tiny_kl_loss_train_sub1=[]
tiny_kl_loss_train_sub2=[]
tiny_kl_loss_test = []
test_fidelity=[]
last_loss = np.zeros(num_clients)  # initial


class Prune():
    def __init__(self, model, percent):
        self.model = model
        self.percent = percent

    def layers_to_prune(self, model):
        # SKIP (will do in pruning function)
        layers = []
        # num_global_weights = 0
        modules = list(model.modules())

        for layer in modules:
            # is_BasicBlock = type(layer) == models.resnet.BasicBlock   #이건 원래 주석처리
            # is_BasicBlock = type(layer) == type(BasicBlock(64, 1))    #model.resnet18 일때 이거해야됨
            is_sequential = type(layer) == nn.Sequential
            is_itself = type(layer) == type(model) if len(modules) > 1 else False

            if (not is_sequential) and (not is_itself) : # and (not is_BasicBlock):
                for name, param in layer.named_parameters():
                    field_name = name.split('.')[-1]
                    # This might break if someone does not adhere to the naming
                    # convention where weights of a module is stored in a field
                    # that has the word 'weight' in it

                    if 'weight' in field_name and param.requires_grad:
                        if field_name.endswith('_orig'):
                            field_name = field_name[:-5]
                        # Might remove the param.requires_grad condition in the future
                        layers.append((layer, field_name))
                        # num_global_weights += torch.numel(param)
        return layers  # , num_global_weights

    def num_weights(self, model):
        # named_weights except bias
        size = 0
        for i in range(len(list(
                model.named_parameters()))):  # for param in parameters 는 bias도 세서 resnet18기준 6000개정도 더많음 (원래 11683712개)
            if 'weight' in list(model.named_parameters())[i][0]:
                size += np.prod(list(list(model.named_parameters())[i][1].shape))
        return size

    def num_parameter_to_prune(self, model, percent):
        # percent= remain_percent_prune
        num_weights = self.num_weights(model)
        num_to_prune = int(num_weights * (1 - (percent / 100)))
        return num_to_prune

    def pruning(self):
        # self.model 을 pruning 함
        num_weights = self.num_weights(self.model)
        layers_to_prune = self.layers_to_prune(self.model)
        num_to_prune = self.num_parameter_to_prune(self.model, self.percent)
        torch.nn.utils.prune.global_unstructured(layers_to_prune, pruning_method=prune.L1Unstructured,
                                                 amount=num_to_prune)

        # remove 하면 named_가 지워져서 prune_result가 안먹히고 prune 한 weight 들이 고정이됨. remove해도 prune 됨
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module,
                                                                                                             nn.Linear):
                prune.remove(module, 'weight')
                try:
                    prune.remove(module, "bias")
                except:
                    pass

        del layers_to_prune
        return self.model

    def copy_model(self, model_ref, model_to_copy):
        new_layers_to_prune, _ = self.layers_to_prune(model_to_copy)
        torch.nn.utils.prune.global_unstructured(new_layers_to_prune,
                                                 pruning_method=torch.nn.utils.prune.L1Unstructured, amount=1)
        for name, params in model_to_copy.named_parameters():
            try:
                params.data.copy_(dict(model_ref.named_parameters())[name])
            except KeyError:
                params.data.copy_(dict(model_ref.named_parameters())[name + '_orig'])
        for name, buffer in model_to_copy.named_buffers():
            buffer.data.copy_(dict(model_ref.named_buffers())[name])

    def prune_result(self, model):
        a = 0
        for i in range(len(list(model.named_parameters()))):
            if 'weight' in list(model.named_parameters())[i][0]:
                a += np.prod(list(list(model.named_parameters())[i][1].shape))

        b = 0
        for i in range(len(list(model.named_buffers()))):
            if 'weight' in list(model.named_buffers())[i][0]:
                b += torch.sum(list(model.named_buffers())[i][1]).item()
        print('original params:', a)
        print('Number of pruned weights (not bias): ', a - b)
        print('remained parameters after pruning: ', b)


# MAIN
print('Training...')
start_time= time.time()

for epoch in range(num_epoch):
    loss_train, kl_loss_train= 0.0, 0.0
    kl_loss_train_sub1, kl_loss_train_sub2= 0.0,0.0
    acc_test_ser, loss_test_ser, acc_test_tiny, loss_test_tiny, kl_loss_test = 0.0, 0.0, 0.0, 0.0, 0.0

    # inters=torch.zeros((num_clients,196, 128, 64, 8, 8))
    if epoch==warmup_epoch:
        last_loss = np.zeros(num_clients)

    if epoch==0:
        # server.load_state_dict(torch.load('./server_weight_init.pth'))
        server.train()
        tiny_server.train()
        for c in range(num_clients):
            clients[c].train()

    for batch_idx, data in enumerate(zip(*trainloader)):

        client_uplink = np.zeros(num_clients)
        try:
            torch.save(server.state_dict(), './server_weight_init.pth') #해야됨 prune 때문에
            torch.save(tiny_server.state_dict(), './tiny_server_weight.pth')  # 해야됨 prune 때문에
        except:
            pass



        for client_idx in range(num_clients):
            inputs, labels = data[int(client_idx)]
            inputs, labels = inputs.to(device), labels.to(device)

            # bb client forward
            client_side_intermidiate = clients[client_idx](inputs)
            intermediate_to_server = client_side_intermidiate.clone().detach().requires_grad_(True)

            # cc Tiny server forward
            if mode=='prunetoteacher' or 'prunetoteacher_loss1':
                tiny_out = tiny_server(intermediate_to_server)
            elif mode=='prunetostudent':
                if client_idx==0:
                    tiny_server2=copy.deepcopy(tiny_server)
                    pruning_model= Prune(tiny_server2, remain_percent_prune)
                    tiny_server_student = pruning_model.pruning()
                tiny_out = tiny_server_student(intermediate_to_server)
            else: #disill
                tiny_out = tiny_server(intermediate_to_server)

            loss_est = criterion(tiny_out, labels)

            # TODO uplink
            if not (last_loss[client_idx]-loss_est.item() >= last_loss[client_idx]*threshold) or epoch<warmup_epoch:
                uplinks[client_idx, epoch * batch_num + batch_idx] = 1

                client_uplink[client_idx] = 1

                opt_c[client_idx].zero_grad()
                opt_s.zero_grad()
                opt_t.zero_grad()

                if mode=='prunetoteacher' or 'prunetoteacher_loss1':
                    # pruning server
                    if client_idx == 0:
                        pruning_model = Prune(server, remain_percent_prune)
                        teacher_server = pruning_model.pruning()
                        # pruning_model.prune_result(server)
                    server.load_state_dict(torch.load('./server_weight_init.pth'))

                # aa server forward
                out_ser = server(intermediate_to_server)
                loss_ser = criterion(out_ser, labels)
                _, preds = torch.max(out_ser.data, 1)
                loss_train += loss_ser.item()

                # aa server backward_1
                loss_ser.backward(retain_graph=True)
                grad_to_client_ser = intermediate_to_server.grad.clone()
                opt_s.step()

                torch.save(server.state_dict(), './server_weight_{}.pth'.format(client_idx))



                # bb client backward
                client_side_intermidiate.backward(grad_to_client_ser)
                opt_c[client_idx].step()
                del client_side_intermidiate


                # cc Tiny server update
                if mode=='prunetoteacher':
                    out_ser_prune = teacher_server(intermediate_to_server)
                    tiny_train_loss_kl = DistillationLoss('soft', alpha, tau)(tiny_out, labels, out_ser_prune)
                elif mode=='prunetostudent':
                    tiny_out_notpruned= tiny_server(intermediate_to_server)
                    tiny_train_loss_kl = DistillationLoss('soft', alpha, tau)(tiny_out_notpruned, labels, out_ser)
                elif mode=='prunetoteacher_loss1':
                    out_ser_prune = teacher_server(intermediate_to_server)
                    try:
                        loss1 = nn.CosineEmbeddingLoss()(tiny_out, tiny_out_ex, torch.Tensor([1]).to(device))
                        loss2 = nn.CosineEmbeddingLoss()(out_ser_prune, out_ser_prune_ex, torch.Tensor([1]).to(device))
                        tiny_train_loss_kl = alpha*torch.nn.L1Loss()(loss1, loss2)*(tau**2) + (1-alpha)*criterion(tiny_out, labels)
                    except: # first iteration, ex no exist
                        tiny_train_loss_kl = DistillationLoss('soft', alpha, tau)(tiny_out, labels, out_ser_prune)
                    tiny_out_ex=copy.deepcopy(tiny_out.detach())
                    out_ser_prune_ex=copy.deepcopy(out_ser_prune.detach())
                elif mode=='prunetoteacher_loss1_kl':
                    out_ser_prune = teacher_server(intermediate_to_server)
                    try:
                        loss1 = DistillationLoss('soft', 1, tau)(tiny_out, labels, tiny_out_ex)
                        loss2 = DistillationLoss('soft', 1, tau)(out_ser_prune, labels, out_ser_prune_ex)
                        tiny_train_loss_kl = alpha*torch.nn.L1Loss()(loss1, loss2)*(tau**2) + (1-alpha)*criterion(tiny_out, labels)
                    except: # first iteration, ex no exist
                        tiny_train_loss_kl = DistillationLoss('soft', alpha, tau)(tiny_out, labels, out_ser_prune)
                    tiny_out_ex=copy.deepcopy(tiny_out.detach())
                    out_ser_prune_ex=copy.deepcopy(out_ser_prune.detach())
                else: #distill
                    tiny_train_loss_kl = DistillationLoss('soft', alpha, tau)(tiny_out, labels, out_ser)

                kl_loss_train += tiny_train_loss_kl.item()
                tiny_train_loss_kl.backward()
                opt_t.step()
                if mode=='prunetoteacher_loss1' or mode=='prunetoteacher_loss1_kl':
                    try:
                        kl_loss_train_sub1 += loss1.item()
                        kl_loss_train_sub2 += loss2.item()
                    except: # first iteration, ex no exist
                        pass




                try:
                    torch.save(tiny_server.state_dict(), './tiny_server_weight.pth')
                except:
                    pass


                server.load_state_dict(torch.load('./server_weight_init.pth'))  # 이걸해야 parallel sl. 한 서버가 동시에 받아서 하는것


            last_loss[client_idx] = loss_est.item()

                # #TODO uplink skip
                # 이전 batch로 다시 업데이트
                # else :
                #     # aa server forward
                #     opt_s.zero_grad()
                #     try:
                #         intermediate_to_server=torch.load('last_activation_{}_{}.pt'.format(client_idx, batch_idx)).to(device)
                #         try_=True
                #     except FileNotFoundError:
                #         try_=False
                #     if try_==True and torch.sum(intermediate_to_server)>0:
                #         out_ser, _ = server(intermediate_to_server)
                #         loss_ser = criterion(out_ser, labels)
                #         _, preds = torch.max(out_ser.data, 1)
                #         loss_train += loss_ser.item()
                #         acc_train += torch.sum(preds == labels.data)
                #         train_losses_ser[client_idx].append(loss_ser.item())
                #
                #         # aa server backward_1
                #         loss_ser.backward(retain_graph=True)
                #         if np.sum(client_uplink) == 0:
                #             for param_sub, param in zip(server_sub.parameters(), server.parameters()):
                #                 param_sub.grad = param.grad
                #         else:
                #             for param_sub, param in zip(server_sub.parameters(), server.parameters()):
                #                 if param_sub.grad == None:
                #                     param_sub.grad = param.grad
                #
                #                 else:
                #                     param_sub.grad += param.grad


            # Federated aggregation FedAvg
            #aa client
            with torch.no_grad():
                w_avg = clients[0].state_dict()
                for key in w_avg.keys():
                    for client_idx in range(num_clients - 1):
                        w = clients[client_idx + 1].state_dict()
                        w_avg[key] += w[key]
                    w_avg[key] = torch.div(w_avg[key], num_clients)
            for client_idx in range(num_clients):
                model_dict = clients[client_idx].state_dict()
                # model_dict.update(w_avg)
                clients[client_idx].load_state_dict(w_avg)
            del model_dict, w_avg

            ##bb server
            with torch.no_grad():
                for first_client_idx in range(num_clients):
                    if client_uplink[first_client_idx]==1:
                        w_avg = torch.load('./server_weight_{}.pth'.format(first_client_idx))
                        break
                for client_idx in range(num_clients - (first_client_idx+1)):
                    if client_uplink[client_idx+first_client_idx+1] ==1:
                        w = torch.load('./server_weight_{}.pth'.format(client_idx+first_client_idx+1))
                        for key in w_avg.keys():
                            w_avg[key] += w[key]

                num_participants=np.sum(client_uplink)
                if num_participants!=0:
                    for key in w_avg.keys():
                        w_avg[key] = torch.div(w_avg[key], num_participants)
                    server.load_state_dict(w_avg)
            try:
                torch.save(server.state_dict(), './server_weight_init.pth')
            except OSError:
                pass



            gc.collect()
        torch.cuda.empty_cache()



    with torch.no_grad():
        agreement, agreement_ex =0, 0
        for batch_i, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            feature = clients[0](inputs)
            out_server = server(feature)
            loss = criterion(out_server, labels)
            _, preds_ser = torch.max(out_server.data, 1)
            loss_test_ser += loss.item()
            acc_test_ser += torch.sum(preds_ser == labels.data)

            if mode=='prunetostudent':
                tiny_server2=copy.deepcopy(tiny_server)
                pruning_model= Prune(tiny_server2, remain_percent_prune)
                tiny_server_student = pruning_model.pruning()

            out_tiny = tiny_server(feature)
            loss = criterion(out_tiny, labels)
            '''
            # top-k (k=3)

            maxk= max((1,3)) #k
            y_resize= labels.view(-1,1)
            _, preds_tiny = out_tiny.topk(maxk, 1, True, True)
            loss_test_tiny += loss.item()
            acc_test_tiny += torch.eq(preds_tiny, y_resize).sum().float().item()
            '''

            # top-1
            _, preds_tiny = torch.max(out_tiny.data, 1)
            loss_test_tiny += loss.item()
            acc_test_tiny += torch.sum(preds_tiny == labels.data)

            agreement += torch.sum(preds_tiny == preds_ser).item()

            loss_kl = DistillationLoss('soft', 1, tau)(out_tiny, labels, out_server)
            kl_loss_test += loss_kl.item()





    acc.append(acc_test_ser.item() / len(testloader.dataset))
    tiny_acc.append(acc_test_tiny.item() / len(testloader.dataset))
    test_losses_ser.append(loss_test_ser / len(testloader.dataset))
    test_losses_tiny.append(loss_test_tiny / len(testloader.dataset))
    tiny_kl_loss_test.append(kl_loss_test / len(testloader.dataset))
    test_fidelity.append(agreement / len(testloader.dataset))
    train_losses_ser.append(loss_train/(len(trainloader[0].dataset)+len(trainloader[1].dataset)+len(trainloader[2].dataset)+len(trainloader[3].dataset)))
    tiny_kl_loss_train.append(kl_loss_train / (len(trainloader[0].dataset) + len(trainloader[1].dataset) + len(trainloader[2].dataset) + len(trainloader[3].dataset)))
    tiny_kl_loss_train_sub1.append(kl_loss_train_sub1 / (len(trainloader[0].dataset) + len(trainloader[1].dataset) + len(trainloader[2].dataset) + len(trainloader[3].dataset)))
    tiny_kl_loss_train_sub2.append(kl_loss_train_sub2 / (len(trainloader[0].dataset) + len(trainloader[1].dataset) + len(trainloader[2].dataset) + len(trainloader[3].dataset)))

    for i in range(num_clients):
        sch_c[i].step()
    sch_s.step()

    _loss_test = loss_test_ser / len(testloader.dataset)
    for c in range(num_clients):
        print(np.sum(uplinks[c, epoch * batch_num:(epoch + 1) * batch_num]))
    print('Epoch {} | acc: {},  tiny_acc: {},  loss_ser: {},  loss_tiny: {},  kl_loss: {}'.format(epoch + 1, acc[-1],
           tiny_acc[-1], test_losses_ser[-1], test_losses_tiny[-1], tiny_kl_loss_test[-1]))


    # NONIID   num_datasplit= 원래나누려던 숫자/num_clients
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/acc.npy', acc)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/tiny_acc.npy', tiny_acc)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/uplinks.npy', uplinks)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/tiny_kl_loss_train.npy', tiny_kl_loss_train)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/tiny_kl_loss_train_sub1.npy', tiny_kl_loss_train_sub1)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/tiny_kl_loss_train_sub2.npy', tiny_kl_loss_train_sub2)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/tiny_kl_loss_test.npy', tiny_kl_loss_test)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/train_loss_ser.npy', train_losses_ser)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/test_loss_ser.npy', test_losses_ser)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/test_loss_tiny.npy', test_losses_tiny)
    np.save('C:/Users/hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/test_fidelity.npy', test_fidelity)


total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

torch.save(clients[0], 'C:/Users\hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/client.pt')
torch.save(tiny_server, 'C:/Users\hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/tinyserver.pt')
torch.save(server, 'C:/Users\hyeli/Dropbox/나메렝/wml/2021-2022/221010_VGG/th/prunetoteacher_loss1/noniid/1/server.pt')



