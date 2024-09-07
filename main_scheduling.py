import torch
import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import torchvision
from torchsummary import summary
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from model import resnet18, teacher_model, student_model, feature_size
from losses import DistillationLoss, SupConLoss
from dataset import batchsize, SplitData, testloader, num_datasplits, new_dataload
import torch.nn.functional as F
import collections
import random
import time
import gc
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#TODO Parameters
total_epochs=200
Temperature = 7
lr = 0.01
lr_c = 0.01
lr_decay = 0.95
num_clients=5
num_classes=10
alpha= 0.5
local_epochs=2
num_best_clients=5




#TODO model
server = teacher_model().to(device)
server = teacher_model().to(device)
client = student_model(Temperature).to(device)
for param in client.parameters():
    param=0.0
server_models= [teacher_model().to(device) for _ in range(num_clients)]
client_models = [student_model(Temperature).to(device) for _ in range(num_clients)]


optimizer_s = optim.SGD(server.parameters(), lr=lr, momentum=0.9)
# scheduler_s = optim.lr_scheduler.LambdaLR(optimizer=optimizer_s, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_epoch=-1, verbose=False)

optimizer_ss = [optim.SGD(model.parameters(), lr=lr_c) for model in server_models]
# scheduler_ss = [optim.lr_scheduler.LambdaLR(optimizer=optimiz, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_epoch=-1, verbose=False) for optimiz in optimizer_ss]

optimizer_c = [optim.Adam(model.parameters(), lr=lr_c) for model in client_models]
# scheduler_c = [optim.lr_scheduler.LambdaLR(optimizer=optimiz, lr_lambda=lambda epoch: lr_decay ** epoch,
#                                           last_epoch=-1, verbose=False) for optimiz in optimizer_c]



#TODO critierion
criterion_KL = DistillationLoss(distillation_type='soft', alpha=alpha, tau=Temperature)
criterion_CE = nn.CrossEntropyLoss()
criterion_CT = SupConLoss(contrast_mode='all', base_temperature=0.07)
# criterion_CT= supervised_nt_xent_loss()   #label, feature 순 .cpu().detach().numpy()


#TODO dataset per clients
new_dataload(True)
clients_dataset1= [SplitData(i,0) for i in range(num_datasplits)]
clients_dataset2= [SplitData(i,1) for i in range(num_datasplits)]
clients_dataset= clients_dataset1+clients_dataset2
# clients_dataset=clients_dataset1
clients_dataloader=[DataLoader(dataset, batch_size=batchsize, shuffle=False) for dataset in clients_dataset]





class Distill:
    def __init__(self):
        pass

    def select_straggler(self):
        num_straggler= 1
        straggler_list= [0]
        client_list= [1,2,3,4,5,6,7,8,9]

        return client_list, straggler_list


    def transforms_images(self, image_tensor):
        device = image_tensor.device
        _, _, H, W = image_tensor.shape

        cr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((H, W)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        image_tensor = torch.cat([cr_transforms(image).unsqueeze_(0)
                                  for image in image_tensor.cpu()], dim=0)

        return image_tensor.to(device)


    def choose_best_clients(self, epoch, client_list, client_acc, number_of_best_clients):
        # choose best client who sends feature to straggler (for CT loss)
        if epoch==0:
            best_client_list= tf.random.shuffle(client_list)[0:number_of_best_clients]
        else:
            list=np.arange(num_clients)
            for i in client_list:
                list[i]= client_acc[i][-1]
            best_client_list= np.argsort(list)[0:number_of_best_clients]

        return best_client_list


    def straggler_update1(self, straggler_idx, best_idx_list, data):
        inputs, labels= data[straggler_idx]
        inputs, labels= inputs.to(device), labels.to(device)
        optimizer_c[straggler_idx].zero_grad()

        # augmentation own data for positive samples
        feature, outputs, out_ct = client_models[straggler_idx](inputs)
        _, _, out_ct_aug = client_models[straggler_idx](self.transforms_images(inputs))
        out2= torch.cat([out_ct, out_ct_aug], 0)
        label2= torch.cat([labels, labels])

        # Positive & negative samples from best clients
        for i in best_idx_list:
            with torch.no_grad():
                inputs_best, labels_best= data[i]
                inputs_best, labels_best = inputs_best.to(device), labels_best.to(device)
                _, _, out_ct_best = client_models[i](inputs_best)

            out2= torch.cat([out2, out_ct_best.detach()],0)
            label2= torch.cat([label2, labels_best])


        # Supcon loss (SimCLR is in Appendix)
        out3 = F.normalize(out2, dim=2)
        loss_ct = criterion_CT(out3, label2)
        loss_ct.backward()

        if torch.isnan(loss_ct)==True:
            print('Warning: loss_ct nan')
        optimizer_c[straggler_idx].step()



    def straggler_update2(self, straggler_idx, best_idx_list, data):
        inputs, labels= data[straggler_idx]
        inputs, labels= inputs.to(device), labels.to(device)
        optimizer_c[straggler_idx].zero_grad()
        loss_mse= nn.MSELoss()
        losses= torch.zeros((len(best_idx_list)))
        losses_idx=0

        # augmentation own data for positive samples
        feature, outputs, out_ct = client_models[straggler_idx](inputs)

        # Positive & negative samples from best clients
        for i in best_idx_list:
            inputs_best, labels_best= data[i]
            inputs_best, labels_best = inputs_best.to(device), labels_best.to(device)
            feature_best, _, out_ct_best = client_models[i](inputs_best)

            losses[losses_idx]= loss_mse(feature, feature_best.detach())#*(len(best_idx_list)+1-losses_idx)
            losses_idx += 1
        loss_m = torch.sum(losses)
        loss_m.backward()
        optimizer_c[straggler_idx].step()


    def mergeGrad(self, modelA, modelB):
        listGradA = []
        listGradB = []
        for pA, pB in zip(modelA.parameters(), modelB.parameters()):
            listGradA.append(pA)
            listGradB.append(pB)
            sum_grad = (pA.grad.clone() + pB.grad.clone())
            pA.grad = sum_grad
            pB.grad = sum_grad

    '''
    def client_aggregate(self, client_list):
        w1 = client_models[client_list[0]].state_dict()
        for client_idx in client_list:
            exec('weight_{} = client_models[client_idx].state_dict()'.format(client_idx))
        for key in eval('weight_{}'.format(client_list[0])):
            exec('weight_{}[key] = weight_{}[key]/len(client_list)'.format(client_list[0], client_list[0]))
            for client_idx in client_list:
                exec('weight_{}[key] += weight_{}[key]/len(client_list)'.format(client_list[0], client_idx))
        for client_idx in client_list:
            client_models[client_idx].load_state_dict(eval('weight_{}'.format(client_list[0])))
    '''

    def client_aggregate(self, client_list):
        with torch.no_grad():
            w_avg= client.state_dict()
            for key in w_avg.keys():
                for client_idx in client_list:
                    w= client_models[client_idx].state_dict()
                    w_avg[key] += w[key]
                w_avg[key] = torch.div(w_avg[key], len(client_list))

            for client_idx in client_list:
                model_dict = client_models[client_idx].state_dict()
                model_dict.update(w_avg)


    def train(self, client_list, epoch, for_fail_straggler_list, back_fail_straggler_list, client_acc, client_loss):

        # client_acc = [[] for _ in range(num_clients)]
        start= time.time()



        #TODO training

        optimizer_s.zero_grad()
        for batch_idx, data in enumerate(zip(*clients_dataloader)):
            for client_idx in client_list:
                inputs, labels= data[int(client_idx)]
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer_c[client_idx].zero_grad()
                optimizer_s.zero_grad()

                #bb forward
                feature, out, out_ct= client_models[client_idx](inputs)
                feature_to_server= feature.detach().clone().requires_grad_()
                output, _= server(feature_to_server)
                #bb server backward
                loss= criterion_CE(output, labels)
                loss.backward()
                #bb client backward
                grad_to_client= feature_to_server.grad.clone()
                feature.backward(grad_to_client)

                optimizer_s.step()
                optimizer_c[client_idx].step()

        #bb weight avg
        self.client_aggregate(client_list)




            #bb straggler update
        for local_epoch in range(local_epochs):
            for batch_idx, data in enumerate(zip(*clients_dataloader)):
                for straggler_idx in straggler_list:
                    best_client_list= self.choose_best_clients(epoch, client_list, client_acc, num_best_clients)
                    # self.straggler_update1(straggler_idx, best_client_list, data)
                    self.straggler_update2(straggler_idx, best_client_list, data)


        end= time.time()





        #TODO test

        c_acc, c_loss = 0.0, 0.0

        for batch_idx, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                feature, out, out_kd= client_models[0](inputs)
                out, _= server(feature)
                loss= criterion_CE(out, labels)
                _, preds= torch.max(out.data, 1)

                c_loss += loss.item()
                #c_acc[num] += preds.eq(labels.view_as(preds)).sum().item()
                c_acc += torch.sum(preds == labels.data)

        print('\nEpoch {}      time: {}sec'.format(epoch, round(end-start,2)))
        with torch.no_grad():
            c_loss_= c_loss/len(testloader.dataset)
            c_acc_= c_acc.double() / len(testloader.dataset)
            client_acc.append(c_acc_)
            print('clients loss: {}, acc: {}'.format(c_loss_, c_acc_))


        return server, client_models, client_acc, client_loss



#TODO main_________________________________________________________________________________________________________

print('Datasets: {} samples per client (Each client has {}samples per class)'.format((50000/num_datasplits), (5000/num_datasplits)))

client_loss = []
client_acc = []
distill=Distill()

for epoch in range(total_epochs):
    # client_list, straggler_list = self.select_straggler()  #communication off

    client_list = np.arange(num_clients)
    straggler_list=[]
    for_fail_straggler_list = []
    back_fail_straggler_list = []

    # if epoch%5!=4:
    #     client_list, straggler_list= self.select_straggler()  # communication-off
    # else:
    #     client_list= np.arange(num_clients)
    #     straggler_list=[]
    server, client_models, client_acc, client_loss= distill.train(client_list, epoch, for_fail_straggler_list, back_fail_straggler_list, client_acc, client_loss)



np.save('D:/Dropbox/나메렝/wml/210722/splitfed_cat.npy', client_acc)
# np.save('D:/Dropbox/나메렝/wml/210715/scheduling_loss.npy', client_loss)










#TODO Appendix
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