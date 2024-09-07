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
import matplotlib.pyplot as plt
import time
import copy
import math
import os
from model import resnet18, teacher_model, student_model
from losses import DistillationLoss
from model import student_model, teacher_model, resnet18, projection_head, tiny_server_model_3
from model_vgg import VGGNet, VGGNet_client, VGGNet_server, VGGNet_tinyserver
from dataset import batchsize, SplitData, num_datasplits, new_dataload, NoniidSplitData
import collections

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)


# Parameters





transform_train = transforms.Compose([
    # #transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Split dataset
# new_dataload(True)
# trainset = SplitData(0,0)

#all dataset
transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)


set_seed(0)
trainloader= DataLoader(trainset, num_workers=0, shuffle=True, batch_size= batchsize, worker_init_fn=np.random.seed(0))

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
set_seed(0)
testloader= DataLoader(testset, num_workers=0, shuffle=True, batch_size= batchsize, worker_init_fn=np.random.seed(0))


criterion= nn.CrossEntropyLoss()
learning_rate=0.1



# client= student_model(7).to(device)   #out 3개
# server= teacher_model().to(device)
# resnet= resnet18().to(device)  #out 1개
# tiny_server= tiny_server_model_3().to(device)
vgg= VGGNet('vggtiny').to(device)
vgg_client=VGGNet_client().to(device)
vgg_server=VGGNet_server().to(device)
vgg_tinyserver= VGGNet_tinyserver().to(device)

# opt_c= optim.SGD(client.parameters(), lr= learning_rate, momentum=0.9)
# opt_s= optim.SGD(server.parameters(), lr= learning_rate, momentum=0.9)
# opt_r= optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9)
# opt_t= optim.SGD(tiny_server.parameters(), lr=learning_rate, momentum=0.9)
opt_v= optim.SGD(vgg.parameters(), lr=learning_rate, momentum=0.9)
opt_vc= optim.SGD(vgg_client.parameters(), lr= learning_rate, momentum=0.9)
opt_vs= optim.SGD(vgg_server.parameters(), lr= learning_rate, momentum=0.9)
opt_vt= optim.SGD(vgg_tinyserver.parameters(), lr=learning_rate, momentum=0.9)


num_epoch=150






'''


testloss_split=np.arange(num_epoch)
testacc_split = np.arange(num_epoch)


tiny_server= tiny_server_model_3().to(device)
opt_t= optim.SGD(tiny_server.parameters(), lr=0.01, momentum=0.9)

print('Entire model training')
for epoch in range(20):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # inputs, labels = inputs.permute(0,3,1,2).to(device), labels.to(device)
        inputs, labels= inputs.to(device), labels.to(device)
        opt_v.zero_grad()
        #bb forward
        out = vgg(inputs)
        #bb loss
        loss= criterion(out, labels)
        loss.backward()
        _, preds= torch.max(out.data, 1)
        loss_train +=loss.item()
        acc_train += torch.sum(preds == labels.data)
        #bb backward
        #bb step
        opt_v.step()



    for batch_i, (inputs, labels) in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        out = vgg(inputs)
        loss = criterion(out, labels)
        _, preds = torch.max(out.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)


    # _loss_train= loss_train/len(dataloader.dataset)
    # trainloss_split[epoch]= _loss_train
    # _acc_train= acc_train/len(trainloader.dataset)
    # trainacc_split[epoch]= _acc_train

    _loss_test = loss_test / len(testloader.dataset)
    testloss_split[epoch] = _loss_test
    _acc_test = acc_test / len(testloader.dataset)
    testacc_split[epoch] = _acc_test
    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))

'''


'''
testloss_split=np.arange(num_epoch)
testacc_split = np.arange(num_epoch)


tiny_server= tiny_server_model_3().to(device)
opt_t= optim.SGD(tiny_server.parameters(), lr=0.01, momentum=0.9)

print('Split No detach_Tiny server local')
for epoch in range(20):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # inputs, labels = inputs.permute(0,3,1,2).to(device), labels.to(device)
        inputs, labels= inputs.to(device), labels.to(device)
        opt_c.zero_grad()
        opt_t.zero_grad()
        #bb forward
        client_side_intermidiate, outputs, out_kd= client(inputs)
        _, _, out = tiny_server(client_side_intermidiate)
        #bb loss
        loss= criterion(out, labels)
        loss.backward()
        _, preds= torch.max(out.data, 1)
        loss_train +=loss.item()
        acc_train += torch.sum(preds == labels.data)
        #bb backward
        #bb step
        opt_c.step()
        opt_t.step()



    for batch_i, (inputs, labels) in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        client_side_intermidiate, outputs, out_kd = client(inputs)
        intermediate_to_server = client_side_intermidiate.detach()
        _, _, out = tiny_server(intermediate_to_server)
        loss = criterion(out, labels)
        _, preds = torch.max(out.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)


    # _loss_train= loss_train/len(dataloader.dataset)
    # trainloss_split[epoch]= _loss_train
    # _acc_train= acc_train/len(trainloader.dataset)
    # trainacc_split[epoch]= _acc_train

    _loss_test = loss_test / len(testloader.dataset)
    testloss_split[epoch] = _loss_test
    _acc_test = acc_test / len(testloader.dataset)
    testacc_split[epoch] = _acc_test
    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))

'''




testloss_split=np.zeros((num_epoch))
testacc_split=np.zeros((num_epoch))


print('Split gradient communicate, no detach')
for epoch in range(num_epoch):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # inputs, labels = inputs.permute(0,3,1,2).to(device), labels.to(device)
        inputs, labels= inputs.to(device), labels.to(device)
        opt_vc.zero_grad()
        opt_vt.zero_grad()
        #bb forward
        client_side_intermidiate = vgg_client(inputs)                   #vgg
        # client_side_intermidiate, outputs, out_kd= client(inputs)     #resnet

        out = vgg_tinyserver(client_side_intermidiate)                        #vgg
        # _,_, out= tiny_server(intermediate_to_server)                 #resnet
        #bb loss
        loss= criterion(out, labels)
        loss.backward()
        _, preds= torch.max(out.data, 1)
        loss_train +=loss.item()
        acc_train += torch.sum(preds == labels.data)
        #bb backward
        #bb step
        opt_vc.step()
        opt_vt.step()



    for batch_i, (inputs, labels) in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        client_side_intermidiate = vgg_client(inputs)                 #vgg
        # client_side_intermidiate, outputs, out_kd = client(inputs)  #resnet
        intermediate_to_server = client_side_intermidiate.detach()
        out = vgg_tinyserver(intermediate_to_server)                  #vgg
        # _,_,out = tiny_server(intermediate_to_server)               #resnet
        loss = criterion(out, labels)
        _, preds = torch.max(out.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)


    # _loss_train= loss_train/len(dataloader.dataset)
    # trainloss_split[epoch]= _loss_train
    # _acc_train= acc_train/len(trainloader.dataset)
    # trainacc_split[epoch]= _acc_train

    _loss_test = loss_test / len(testloader.dataset)
    testloss_split[epoch] = _loss_test
    _acc_test = acc_test / len(testloader.dataset)
    testacc_split[epoch] = _acc_test
    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))

plt.subplot(1, 2, 1)
plt.plot(testloss_split)
plt.subplot(1, 2, 2)
plt.plot(testacc_split)
plt.show()







testloss_split=np.zeros((num_epoch))
testacc_split=np.zeros((num_epoch))


print('Split gradient communicate, detach_ Tiny server local')
for epoch in range(num_epoch):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # inputs, labels = inputs.permute(0,3,1,2).to(device), labels.to(device)
        inputs, labels= inputs.to(device), labels.to(device)
        opt_vc.zero_grad()
        opt_vt.zero_grad()
        #bb forward
        client_side_intermidiate = vgg_client(inputs)                   #vgg
        # client_side_intermidiate, outputs, out_kd= client(inputs)     #resnet
        intermediate_to_server= client_side_intermidiate.detach().requires_grad_()
        out = vgg_tinyserver(intermediate_to_server)                        #vgg
        # _,_, out= tiny_server(intermediate_to_server)                 #resnet
        #bb loss
        loss= criterion(out, labels)
        loss.backward()
        _, preds= torch.max(out.data, 1)
        loss_train +=loss.item()
        acc_train += torch.sum(preds == labels.data)
        #bb backward
        grad_to_client= intermediate_to_server.grad.clone()
        client_side_intermidiate.backward(grad_to_client)
        #bb step
        opt_vc.step()
        opt_vt.step()



    for batch_i, (inputs, labels) in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        client_side_intermidiate = vgg_client(inputs)                 #vgg
        # client_side_intermidiate, outputs, out_kd = client(inputs)  #resnet
        intermediate_to_server = client_side_intermidiate.detach()
        out = vgg_tinyserver(intermediate_to_server)                  #vgg
        # _,_,out = tiny_server(intermediate_to_server)               #resnet
        loss = criterion(out, labels)
        _, preds = torch.max(out.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)


    # _loss_train= loss_train/len(dataloader.dataset)
    # trainloss_split[epoch]= _loss_train
    # _acc_train= acc_train/len(trainloader.dataset)
    # trainacc_split[epoch]= _acc_train

    _loss_test = loss_test / len(testloader.dataset)
    testloss_split[epoch] = _loss_test
    _acc_test = acc_test / len(testloader.dataset)
    testacc_split[epoch] = _acc_test
    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))

plt.subplot(1, 2, 1)
plt.plot(testloss_split)
plt.subplot(1, 2, 2)
plt.plot(testacc_split)
plt.show()





'''
testloss_split=np.zeros((num_epoch))
testacc_split=np.zeros((num_epoch))

alpha=0.4
tau=2
distill_loss= DistillationLoss('soft', alpha=alpha, tau=tau)

server_teacher=torch.load('D:/Dropbox/나메렝/wml/221010_VGG/final_server_model.pth')


print('Split gradient communicate, Distillation')
for epoch in range(num_epoch):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # inputs, labels = inputs.permute(0,3,1,2).to(device), labels.to(device)
        inputs, labels= inputs.to(device), labels.to(device)
        opt_vc.zero_grad()
        opt_vt.zero_grad()
        #bb forward
        client_side_intermidiate = vgg_client(inputs)                   #vgg
        # client_side_intermidiate, outputs, out_kd= client(inputs)     #resnet
        intermediate_to_server= client_side_intermidiate.detach().requires_grad_()
        teacher_knowledge = server_teacher(intermediate_to_server)
        out = vgg_tinyserver(intermediate_to_server)                        #vgg
        # _,_, out= tiny_server(intermediate_to_server)                 #resnet
        #bb loss
        loss= distill_loss(out, labels, teacher_knowledge)
        loss.backward()
        _, preds= torch.max(out.data, 1)
        loss_train +=loss.item()
        acc_train += torch.sum(preds == labels.data)
        #bb backward
        grad_to_client= intermediate_to_server.grad.clone()
        client_side_intermidiate.backward(grad_to_client)
        #bb step
        opt_vc.step()
        opt_vt.step()



    for batch_i, (inputs, labels) in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        client_side_intermidiate = vgg_client(inputs)                 #vgg
        # client_side_intermidiate, outputs, out_kd = client(inputs)  #resnet
        intermediate_to_server = client_side_intermidiate.detach()
        out = vgg_tinyserver(intermediate_to_server)                  #vgg
        # _,_,out = tiny_server(intermediate_to_server)               #resnet
        loss = criterion(out, labels)
        _, preds = torch.max(out.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)


    # _loss_train= loss_train/len(dataloader.dataset)
    # trainloss_split[epoch]= _loss_train
    # _acc_train= acc_train/len(trainloader.dataset)
    # trainacc_split[epoch]= _acc_train

    _loss_test = loss_test / len(testloader.dataset)
    testloss_split[epoch] = _loss_test
    _acc_test = acc_test / len(testloader.dataset)
    testacc_split[epoch] = _acc_test
    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))

    if epoch%5==0 or epoch==num_epoch-1:
        np.save('D:/Dropbox/나메렝/wml/221010_VGG/test_acc_tinydistill.npy', testacc_split)
        np.save('D:/Dropbox/나메렝/wml/221010_VGG/test_loss_tinydistill.npy', testloss_split)

plt.subplot(1, 2, 1)
plt.plot(testloss_split)
plt.subplot(1, 2, 2)
plt.plot(testacc_split)
plt.show()
'''

# torch.save(vgg_client, 'D:/Dropbox/나메렝/wml/221010_VGG/final_client_model.pth')
# torch.save(vgg_server, 'D:/Dropbox/나메렝/wml/221010_VGG/final_server_model.pth')





'''
testloss_split=np.zeros((num_epoch))
testacc_split=np.zeros((num_epoch))

alpha=0.4
tau=2
distill_loss= DistillationLoss('soft', alpha=alpha, tau=tau)

server_teacher=torch.load('D:/Dropbox/나메렝/wml/221010_VGG/final_server_model.pth')
client=torch.load('D:/Dropbox/나메렝/wml/221010_VGG/final_client_model.pth')


print('Fixed client and server, Distillation')
for epoch in range(num_epoch):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # inputs, labels = inputs.permute(0,3,1,2).to(device), labels.to(device)
        inputs, labels= inputs.to(device), labels.to(device)
        opt_vt.zero_grad()
        #bb forward
        client_side_intermidiate = client(inputs)                   #vgg
        # client_side_intermidiate, outputs, out_kd= client(inputs)     #resnet
        intermediate_to_server= client_side_intermidiate.detach().requires_grad_()
        teacher_knowledge = server_teacher(intermediate_to_server)
        out = vgg_tinyserver(intermediate_to_server)                        #vgg
        # _,_, out= tiny_server(intermediate_to_server)                 #resnet
        #bb loss
        loss= distill_loss(out, labels, teacher_knowledge)
        loss.backward()
        _, preds= torch.max(out.data, 1)
        loss_train +=loss.item()
        acc_train += torch.sum(preds == labels.data)
        #bb backward
        grad_to_client= intermediate_to_server.grad.clone()
        client_side_intermidiate.backward(grad_to_client)
        #bb step
        opt_vt.step()



    for batch_i, (inputs, labels) in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        client_side_intermidiate = vgg_client(inputs)                 #vgg
        # client_side_intermidiate, outputs, out_kd = client(inputs)  #resnet
        intermediate_to_server = client_side_intermidiate.detach()
        out = vgg_tinyserver(intermediate_to_server)                  #vgg
        # _,_,out = tiny_server(intermediate_to_server)               #resnet
        loss = criterion(out, labels)
        _, preds = torch.max(out.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)


    # _loss_train= loss_train/len(dataloader.dataset)
    # trainloss_split[epoch]= _loss_train
    # _acc_train= acc_train/len(trainloader.dataset)
    # trainacc_split[epoch]= _acc_train

    _loss_test = loss_test / len(testloader.dataset)
    testloss_split[epoch] = _loss_test
    _acc_test = acc_test / len(testloader.dataset)
    testacc_split[epoch] = _acc_test
    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))

    if epoch%5==0 or epoch==num_epoch-1:
        np.save('D:/Dropbox/나메렝/wml/221010_VGG/test_acc_tinydistill_clifixed.npy', testacc_split)
        np.save('D:/Dropbox/나메렝/wml/221010_VGG/test_loss_tinydistill_clifixed.npy', testloss_split)

plt.subplot(1, 2, 1)
plt.plot(testloss_split)
plt.subplot(1, 2, 2)
plt.plot(testacc_split)
plt.show()
'''



'''

print('Split gradient communicate')
for epoch in range(num_epoch):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader,0):
        inputs, labels = inputs.to(device), labels.to(device)
        opt_c.zero_grad()
        opt_s.zero_grad()
        #bb forward
        client_side_intermidiate, outputs, out_kd= client(inputs)
        intermediate_to_server= client_side_intermidiate.clone()
        out, out_= server(intermediate_to_server)
        #bb loss
        loss= criterion(out, labels)
        loss.backward()  #이거 해도 None  retain_grad() 하고 이거 하면 1 로 grad 출력함

        _, preds= torch.max(out.data, 1)
        loss_train +=loss.item()
        acc_train += torch.sum(preds == labels.data)

        #bb step
        opt_c.step()
        opt_s.step()

    for batch_i, (inputs, labels) in enumerate(testloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        client_side_intermidiate, outputs, out_kd = client(inputs)
        intermediate_to_server = client_side_intermidiate.detach()
        out, out_ = server(intermediate_to_server)
        loss = criterion(out, labels)
        _, preds = torch.max(out.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data)

    _loss_train = loss_train / len(trainloader.dataset)
    trainloss_split[epoch] = _loss_train
    _acc_train = acc_train / len(trainloader.dataset)
    trainacc_split[epoch] = _acc_train

    _loss_test = loss_test / len(testloader.dataset)
    testloss_split[epoch] = _loss_test
    _acc_test = acc_test / len(testloader.dataset)
    testacc_split[epoch] = _acc_test
    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))


new_dataload()
dataset= SplitData(0,0)
dataloader=DataLoader(dataset, batch_size=100, shuffle=False)
'''


'''


new_dataload()
trainset= SplitData(0,0)
trainloader= DataLoader(trainset, shuffle=False, batch_size=100)

print('\nResNet18 alone')
acc=[]
lossss=[]
resnet1= resnet18().to(device)
resnet2= resnet18().to(device)

resnet2.load_state_dict(resnet1.state_dict()) #hook 안걸림

opt_r1= optim.SGD(resnet1.parameters(), lr=0.01, momentum=0.9)
opt_r2= optim.SGD(resnet2.parameters(), lr=0.01, momentum=0.9)

for epoch in range(20):
    loss_test1, acc_test1, loss_test2, acc_test2 = 0.0, 0.0, 0.0, 0.0
    weight= resnet2.state_dict()
    for batch_idx, (data1, data2) in enumerate(zip(trainloader, trainloader)):
        resnet1.load_state_dict(weight)

        inputs1, labels1 = data1
        inputs2, labels2 = data2
        inputs1, labels1 = inputs1.to(device), labels1.to(device)
        inputs2, labels2 = inputs2.to(device), labels2.to(device)
        opt_r1.zero_grad()
        opt_r2.zero_grad()
        outputs1 = resnet1(inputs1)
        # outputs2 = resnet2(inputs2)

        loss1 = criterion(outputs1, labels1)
        # loss2 = criterion(outputs2, labels2)

        loss1.backward()
        # loss2.backward()

        for p1, p2 in zip(resnet1.parameters(), resnet2.parameters()):
            # p2.grad = (p1.grad + p1.grad)/2
            p2.grad=p1.grad

        opt_r1.step()
        opt_r2.step()


    for batch_i, (data1, data2) in enumerate(zip(testloader, testloader)):
        with torch.no_grad():
            input1, label1 = data1
            input2, label2 = data2
            input1, label1 = input1.to(device), label1.to(device)
            input2, label2 = input2.to(device), label2.to(device)
            outputs1 = resnet1(input1)
            loss1= criterion(outputs1, label1)
            _, preds_s1= torch.max(outputs1.data, 1)

            loss_test1 += loss1.item()
            acc_test1 += torch.sum(preds_s1 == label1.data)


            outputs2 = resnet2(input2)
            loss2 = criterion(outputs2, label2)
            _, preds_s2 = torch.max(outputs2.data, 1)

            loss_test2 += loss2.item()
            acc_test2 += torch.sum(preds_s2 == label2.data)

    _loss_test1 = loss_test1 / len(testloader.dataset)
    _acc_test1 = acc_test1 / len(testloader.dataset)
    _loss_test2 = loss_test2 / len(testloader.dataset)
    _acc_test2 = acc_test2 / len(testloader.dataset)

    print('Epoch {}\nrenset1:    loss: {}, acc: {}\nrenset2:    loss: {}, acc: {}'.format(epoch, _loss_test1, _acc_test1, _loss_test2, _acc_test2))









client2= student_model(7).to(device)
client3= student_model(7).to(device)


epoch=200

# activation
diff_acts_norm=[]
diff_acts_cos=[]
samedata_acts_norm=[]
samedata_acts_cos=[]
# data
diff_data_label=[]
diff_data_qual=[]
# weight
diff_weight_grad=[]
diff_weight_norm=[]
#다 batch마다 측정해서 epoch 으로 평균



print('\nSplit learning 1client 1server')
criterion= nn.CrossEntropyLoss()
lossss=[]
accss=[]

# activation
diff_acts_norm_batch = []
diff_acts_cos_batch = []
samedata_acts_norm_batch = []
samedata_acts_cos_batch = []
# data
diff_data_label_batch = []
diff_data_qual_batch = []
# weight
diff_weight_grad_batch = []
diff_weight_norm_batch = []
#acc
accss_batch=[]


for epoch in range(epoch):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0

    # # activation
    # diff_acts_norm_batch = []
    # diff_acts_cos_batch = []
    # samedata_acts_norm_batch = []
    # samedata_acts_cos_batch = []
    # # data
    # diff_data_label_batch = []
    # diff_data_qual_batch = []
    # # weight
    # diff_weight_grad_batch = []
    # diff_weight_norm_batch = []

    c=0

    for batch_idx, data in enumerate(zip(trainloader, testloader)):
        #bb train
        if batch_idx>0:
            prev_activation=activation
            prev_activation_cosine= activation_cosine
            prev_labels= labels
            prev_grad_to_client= grad_to_client
            # prev_weight= copy.deepcopy(weight)
        inputs, labels= data[0]
        inputs, labels = inputs.to(device), labels.to(device)
        if c==0:
            image=inputs
            truth=labels
            c=1
        opt_c.zero_grad()
        opt_s.zero_grad()
        activation, cli_out, activation_cosine = client(inputs)

        inter_to_server= activation.detach().requires_grad_()

        #activation
        if batch_idx > 0:
            diff_acts_norm_batch.append(torch.norm((activation - prev_activation), 'fro').item())
            # cos = torch.nn.CosineEmbeddingLoss()
            # diff_acts_cos_batch.append(cos(activation_cosine, prev_activation_cosine, labels).item())
        #data
        # if batch_idx > 0:
        #     c=0
        #     for la in range(10):
        #         a= torch.where(labels==la, torch.ones_like(labels), torch.zeros_like(labels))
        #         a= torch.sum(a).item()
        #         b = torch.where(prev_labels == la, torch.ones_like(prev_labels), torch.zeros_like(prev_labels))
        #         b = torch.sum(b).item()
        #         c+=abs(a-b)
        #     diff_data_label_batch.append(c)

        outputs, _ = server(inter_to_server)
        loss1 = criterion(outputs, labels)
        loss1.backward()
        grad_to_client= inter_to_server.grad.clone()
        activation.backward(grad_to_client)
        opt_c.step()
        opt_s.step()
        #weight
        # weight = client.state_dict()
        # if batch_idx > 0:
        #     diff_weight_grad_batch.append(torch.norm((grad_to_client - prev_grad_to_client), 'fro').item())
        #
        # if batch_idx > 0:
        #     with torch.no_grad():
        #         client2.load_state_dict(weight)
        #         client3.load_state_dict(prev_weight)
        #     summ = 0.0
        #
        #     for params1, params2 in zip(client2.parameters(), client3.parameters()):
        #         summ += torch.norm((params1 - params2), 'fro')
        #     diff_weight_norm_batch.append(summ.item())
            #samedata
            # act2, act2_cos, _= client2(image)
            # act3, act3_cos, _ = client3(image)
            # samedata_acts_norm_batch.append(torch.norm((act2 - act3), 'fro').item())
            # samedata_acts_cos_batch.append(cos(act2_cos, act3_cos, truth).item())


        #bb test
        input, label= data[1]
        input, label = input.to(device), label.to(device)
        act, _, _ = client(input)
        outputs, _ = server(act)
        loss= criterion(outputs, label)
        _, preds_s= torch.max(outputs.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds_s == label.data)
        accss_batch.append(torch.sum(preds_s == label.data).item())

        #data
        if batch_idx > 0:
            _, preds_cli = torch.max(cli_out.data, 1)
            acc_cli = torch.sum(preds_cli == labels.data)
            # diff_data_qual_batch.append(abs(loss1.item()-loss.item()))
            diff_data_qual_batch.append(acc_cli.item())   #acc_test

    _loss_test = loss_test / len(testloader.dataset)
    _acc_test = acc_test / len(testloader.dataset)
    lossss.append(_loss_test)
    accss.append(_acc_test.item())

    #activation
    # diff_acts_norm.append(np.mean(diff_acts_norm_batch))
    # diff_acts_cos.append(np.mean(diff_acts_cos_batch))
    # # data
    # diff_data_label.append(np.mean(diff_data_label_batch))
    # diff_data_qual.append(np.mean(diff_data_qual_batch))
    # weight
    # diff_weight_grad.append(np.mean(diff_weight_grad_batch))
    # diff_weight_norm.append(np.mean(diff_weight_norm_batch))
    # samedata_acts_norm.append(np.mean(samedata_acts_norm_batch))
    # samedata_acts_cos.append(np.mean(samedata_acts_cos_batch))

    print('Epoch {}    loss: {}, acc: {}'.format(epoch, _loss_test, _acc_test))
    np.save('D:/Dropbox/나메렝/wml/210925/0928/diff_acts_norm_batch.npy', diff_acts_norm_batch)
    # np.save('D:/Dropbox/나메렝/wml/210925/diff_acts_cos_batch.npy', diff_acts_cos_batch)
    # np.save('D:/Dropbox/나메렝/wml/210925/diff_data_label_batch.npy', diff_data_label_batch)
    np.save('D:/Dropbox/나메렝/wml/210925/0928/diff_data_qual_batch.npy', diff_data_qual_batch)
    # np.save('D:/Dropbox/나메렝/wml/210925/0928/diff_weight_grad.npy', diff_weight_grad)
    # np.save('D:/Dropbox/나메렝/wml/210925/0928/diff_weight_norm.npy', diff_weight_norm)
    np.save('D:/Dropbox/나메렝/wml/210925/0928/diff_eval_acc_batch.npy', accss_batch)
    # np.save('D:/Dropbox/나메렝/wml/210925/diff_eval_loss_batch.npy', lossss)
    # np.save('D:/Dropbox/나메렝/wml/210925/samedata_acts_norm_batch.npy', samedata_acts_norm_batch)
    # np.save('D:/Dropbox/나메렝/wml/210925/samedata_acts_cos_batch.npy', samedata_acts_cos_batch)




















epoch=200


print('\nSplit learning 1client 1server local parallel')
metric='loss'
th=0

criterion= nn.CrossEntropyLoss()
loss_s=[]
loss_c=[]
acc_s=[]
acc_c=[]
test_loss=[]
test_acc=[]
opt_c2= optim.SGD(client.parameters(), lr= 0.00001, momentum=0.9)

if metric=='send':
    diff_cli_batch = []
    diff_ser_batch = []
elif metric=='weight':
    diff_cli_batch = []
    diff_ser_batch = []
    client2=student_model(7).to(device)
    client3 = student_model(7).to(device)
    server2= teacher_model().to(device)
    server3 = teacher_model().to(device)
    weight_c= client.state_dict()
    weight_s= server.state_dict()
elif metric=='loss':
    diff_cli_batch=[]
    diff_ser_batch = []
elif metric=='grad':
    diff_cli_batch=[]
    diff_ser_batch=[]


ser_th=0
cli_th=0

# 전송 하면 1
forward_cli=np.zeros(int((50000/(num_datasplits*batchsize))*epoch))
backward_ser=np.zeros(int((50000/(num_datasplits*batchsize))*epoch))
a=0

for epoch in range(epoch):
    acc_train, loss_train = 0.0, 0.0
    acc_test, loss_test = 0.0, 0.0


    if epoch==0:
        if metric=='send':
            prev_act_c = torch.zeros(([len(trainloader)] + [batchsize] + [64, 8, 8])).to(device)
            prev_grad_s = torch.zeros(([len(trainloader)] + [batchsize] + [64, 8, 8])).to(device)
        elif metric=='weight':
            prev_weight_c= copy.deepcopy(weight_c)
            prev_weight_s = copy.deepcopy(weight_s)
        elif metric=='loss':
            prev_loss_c = 0.0
            prev_loss_s = 0.0
        elif metric=='grad':
            prev_grad_c = torch.zeros(([len(trainloader)] + [batchsize] + [64, 8, 8])).to(device)
            prev_grad_s = torch.zeros(([len(trainloader)] + [batchsize] + [64, 8, 8])).to(device)

    cli_send=False
    ser_send=False

    for batch_idx, data in enumerate(trainloader):
        #bb train

        inputs, labels= data
        inputs, labels = inputs.to(device), labels.to(device)


        #aa local parallel
        opt_c2.zero_grad()
        opt_p.zero_grad()

        activation_pr, _, _ = client(inputs)
        inter_to_pr= activation_pr.detach().requires_grad_()
        _, cli_out= pr_head(inter_to_pr)
        loss2 = criterion(cli_out, labels)

        _, preds_c = torch.max(cli_out.data, 1)
        loss_c.append(loss2.item()/ batchsize)
        acc_c.append(torch.sum(preds_c == labels.data).item()/ batchsize)

        loss2.backward()

        grad_pr_to_client= inter_to_pr.grad.clone()

        if metric=='grad':
            diff_cli_batch.append(torch.norm((grad_pr_to_client - prev_grad_c[batch_idx]), 'fro').item())
            if len(diff_cli_batch)<=int(50000/(num_datasplits*batchsize)) or diff_cli_batch[-1] - 1.32*diff_cli_batch[-1-int(50000/(num_datasplits*batchsize))] >cli_th:
                prev_grad_c[batch_idx]= inter_to_pr.grad.clone()
                cli_send=True

        activation_pr.backward(grad_pr_to_client)
        opt_p.step()
        opt_c2.step()

        if metric=='send':
            #local 업데이트 하고 실제 보낼 activation 으로 재야해서
            with torch.no_grad():
                activation, _, _ = client(inputs)
                diff_cli_batch.append(torch.norm((activation - prev_act_c[batch_idx]), 'fro').item())
                if len(diff_cli_batch) <= int(50000 / (num_datasplits * batchsize)) or diff_cli_batch[-1] - 1.03*diff_cli_batch[-1 - int(50000 / (num_datasplits * batchsize))] >cli_th:
                    prev_act_c[batch_idx] = activation
                    cli_send = True
        elif metric=='weight':
            weight_c= client.state_dict()
            with torch.no_grad():
                client2.load_state_dict(weight_c)
                client3.load_state_dict(prev_weight_c)
            summ=0.0
            for params1, params2 in zip(client2.parameters(), client3.parameters()):
                summ+= torch.norm((params1-params2), 'fro')
            diff_cli_batch.append(summ.item())
            if len(diff_cli_batch) <= int(50000 / (num_datasplits * batchsize)) or diff_cli_batch[-1] -5*diff_cli_batch[-1 - int(50000 / (num_datasplits * batchsize))] > cli_th:
                prev_weight_c= copy.deepcopy(weight_c)
                cli_send=True
        elif metric=='loss':
            with torch.no_grad():
                diff_cli_batch.append(abs(loss2-prev_loss_c))
                if len(diff_cli_batch) <= int(50000 / (num_datasplits * batchsize)) or diff_cli_batch[-1] - 8*diff_cli_batch[-1 - int(50000 / (num_datasplits * batchsize))] >cli_th:
                    prev_loss_c = loss2
                    cli_send = True

        # cli_send=True

        if cli_send==True:
            #aa joint update
            opt_c.zero_grad()
            opt_s.zero_grad()

            activation, _, _ = client(inputs)
            inter_to_server= activation.detach().requires_grad_()
            outputs, _ = server(inter_to_server)
            loss1 = criterion(outputs, labels)
            loss1.backward()

            _, preds_s = torch.max(outputs.data, 1)
            loss_s.append(loss1.item()/batchsize)
            acc_s.append(torch.sum(preds_s == labels.data).item()/ batchsize)

            grad_to_client= inter_to_server.grad.clone()

            if metric=='send':
                diff_ser_batch.append(torch.norm((grad_to_client - prev_grad_s[batch_idx]), 'fro').item())
                if len(diff_ser_batch)<=int(50000/(num_datasplits*batchsize)) or diff_ser_batch[-1]-1.1*diff_ser_batch[-1-int(50000/(num_datasplits*batchsize))] >ser_th:
                    prev_grad_s[batch_idx] = inter_to_server.grad.clone()
                    ser_send=True
                    activation.backward(grad_to_client)
                    opt_c.step()

            elif metric=='weight':
                weight_s = server.state_dict()
                with torch.no_grad():
                    server2.load_state_dict(weight_s)
                    server3.load_state_dict(prev_weight_s)
                summ = 0.0
                for params1, params2 in zip(server2.parameters(), server3.parameters()):
                    summ += torch.norm((params1 - params2), 'fro')
                diff_ser_batch.append(summ.item())
                if len(diff_ser_batch) <= int(50000 / (num_datasplits * batchsize)) or diff_ser_batch[-1] - 4*diff_ser_batch[-1 - int(50000 / (num_datasplits * batchsize))] > ser_th:
                    prev_weight_s = copy.deepcopy(weight_s)
                    ser_send = True
                    opt_c.step()
            elif metric=='loss':
                with torch.no_grad():
                    diff_ser_batch.append(abs(loss1-prev_loss_s))
                    # diff_ser_batch.append(loss1 - prev_loss_s)
                    # diff_ser_batch.append(prev_loss_s - loss1)
                    if len(diff_ser_batch)<=int(50000/(num_datasplits*batchsize)) or diff_ser_batch[-1]-20*diff_ser_batch[-1-int(50000/(num_datasplits*batchsize))] >ser_th:
                        prev_loss_s = loss1
                        ser_send=True
                        activation.backward(grad_to_client)
                        opt_c.step()
            elif metric=='grad':
                diff_ser_batch.append(torch.norm((grad_to_client - prev_grad_s[batch_idx]), 'fro').item())
                if len(diff_ser_batch)<=int(50000/(num_datasplits*batchsize)) or diff_ser_batch[-1]-1.1*diff_ser_batch[-1-int(50000/(num_datasplits*batchsize))] >ser_th:
                    prev_grad_s[batch_idx] = inter_to_server.grad.clone()
                    ser_send=True
                    activation.backward(grad_to_client)
                    opt_c.step()
            opt_s.step()

        if cli_send==True:
            forward_cli[a]=1
        if ser_send==True:
            backward_ser[a]=1

        a+=1


    print('Epoch {}   last batch'.format(epoch))
    print('client local loss: {}, acc: {}'.format(loss_c[-1], acc_c[-1]))
    print('server joint loss: {}, acc: {}'.format(loss_s[-1], acc_s[-1]))

    loss_test=0.0
    acc_test=0.0

    for batch_idx, data in enumerate(testloader):
        input, label = data
        input, label = input.to(device), label.to(device)
        act, _, _ = client(input)
        outputs, _ = server(act)
        loss = criterion(outputs, label)
        _, preds_s = torch.max(outputs.data, 1)
        loss_test += loss.item()
        acc_test += torch.sum(preds_s == label.data)

    _loss_test = loss_test / len(testloader.dataset)
    _acc_test = acc_test / len(testloader.dataset)
    test_loss.append(_loss_test)
    test_acc.append(_acc_test.item())


path1='D:/Dropbox/나메렝/wml/211008/'
path2='/th2_mul'
num='8_20'
who='/'#'/client/'

np.save(path1+str(metric)+path2+num+who+'diff_cli_batch.npy', diff_cli_batch)
np.save(path1+str(metric)+path2+num+who+'diff_ser_batch.npy', diff_ser_batch)
np.save(path1+str(metric)+path2+num+who+'forward_cli.npy', forward_cli)
np.save(path1+str(metric)+path2+num+who+'backward_ser.npy', backward_ser)
np.save(path1+str(metric)+path2+num+who+'loss_s.npy', loss_s)
np.save(path1+str(metric)+path2+num+who+'loss_c.npy', loss_c)
np.save(path1+str(metric)+path2+num+who+'acc_s.npy', acc_s)
np.save(path1+str(metric)+path2+num+who+'acc_c.npy', acc_c)
np.save(path1+str(metric)+path2+num+who+'test_loss.npy', test_loss)
np.save(path1+str(metric)+path2+num+who+'test_acc.npy', test_acc)

'''

'''

print('\nResNet18 alone 2 clients')

data1= SplitData(0)
data2= SplitData(1)
dataloader_1= DataLoader(data1, batch_size=50, shuffle=True)
dataloader_2= DataLoader(data2, batch_size=50, shuffle=True)

resnet1= resnet18().to(device)
opt1= optim.SGD(resnet1.parameters(), lr= 0.001, momentum=0.9)
for param in resnet1.parameters():
    param=torch.ones_like(param)*0.01
resnet2= resnet18().to(device)
opt2= optim.SGD(resnet2.parameters(), lr= 0.001, momentum=0.9)
for param in resnet2.parameters():
    param=torch.ones_like(param)*0.01

server1= teacher_model().to(device)
opt1s= optim.SGD(server1.parameters(), lr= 0.001, momentum=0.9)
for param in server1.parameters():
    param=torch.ones_like(param)*0.01
server2= teacher_model().to(device)
opt2s= optim.SGD(server2.parameters(), lr= 0.001, momentum=0.9)
for param in server2.parameters():
    param=torch.ones_like(param)*0.01

client1= student_model(7).to(device)
opt1c= optim.SGD(resnet1.parameters(), lr= 0.001, momentum=0.9)
for param in client1.parameters():
    param=torch.ones_like(param)*0.01
client2= student_model(7).to(device)
opt2c= optim.SGD(client2.parameters(), lr= 0.001, momentum=0.9)
for param in client2.parameters():
    param=torch.ones_like(param)*0.01


trainloss_res1, trainloss_res2 = np.zeros(5), np.zeros(5)
trainacc_res1, trainacc_res2= np.zeros(5), np.zeros(5)

testloss_res1, testloss_res2 = np.zeros(5), np.zeros(5)
testacc_res1, testacc_res2= np.zeros(5), np.zeros(5)

for epoch in range(5):
    acc_train1, acc_train2, loss_train1,  loss_train2 = 0.0, 0.0, 0.0, 0.0
    acc_test1, loss_test1, acc_test2, loss_test2 = 0.0, 0.0, 0.0, 0.0
    for batch_idx, (data1, data2) in enumerate(zip(dataloader_1, dataloader_2)):
        inputs1, labels1= data1
        inputs2, labels2 = data2
        inputs1, labels1 = inputs1.to(device).permute(0, 3, 1, 2).float(), labels1.to(device)
        inputs2, labels2 = inputs2.to(device).permute(0, 3, 1, 2).float(), labels2.to(device)
        opt1.zero_grad()
        opt2.zero_grad()
        outputs1 = resnet1(inputs1).requires_grad_()
        outputs2= resnet2(inputs2).requires_grad_()
        output= (outputs1+outputs2)/2

        output= output.requires_grad_()
        label= ((labels1+labels2)/2).long()
        loss= criterion(output, labels2)
        # loss1 = criterion(outputs1, labels1)
        # loss2 = criterion(outputs2, labels2)

        # _, preds_s1 = torch.max(outputs1.data, 1)
        # _, preds_s2 = torch.max(outputs2.data, 1)
        #
        # loss_train1 += loss1.item()
        # acc_train1 += torch.sum(preds_s1 == labels1.data)
        # loss_train2 += loss2.item()
        # acc_train2 += torch.sum(preds_s2 == labels2.data)

        # loss1.backward()
        # loss2.backward()
        loss.backward()

        for (param1, param2) in zip(resnet1.parameters(), resnet2.parameters()):
            grad= param1.grad + param2.grad
            param1.grad= grad
            param2.grad= grad

        opt1.step()
        opt2.step()



    for batch_i, (input, label) in enumerate(testloader, 0):
        input, label = input.to(device), label.to(device)
        outputs1 = resnet1(input)
        outputs2 = resnet2(input)
        loss1= criterion(outputs1, label)
        loss2 = criterion(outputs2, label)
        _, preds_s1= torch.max(outputs1.data, 1)
        _, preds_s2 = torch.max(outputs2.data, 1)

        loss_test1 += loss1.item()
        acc_test1 += torch.sum(preds_s1 == label.data)
        loss_test2 += loss2.item()
        acc_test2 += torch.sum(preds_s2 == label.data)

    _loss_train1 = loss_train1 / len(trainloader.dataset)
    trainloss_res1[epoch] = _loss_train1
    _acc_train1 = acc_train1 / len(trainloader.dataset)
    trainacc_res1[epoch] = _acc_train1

    _loss_train2 = loss_train2 / len(trainloader.dataset)
    trainloss_res2[epoch] = _loss_train2
    _acc_train2 = acc_train2 / len(trainloader.dataset)
    trainacc_res2[epoch] = _acc_train2

    _loss_test1 = loss_test1 / len(testloader.dataset)
    testloss_res1[epoch] = _loss_test1
    _acc_test1 = acc_test1 / len(testloader.dataset)
    testacc_res1[epoch] = _acc_test1

    _loss_test2 = loss_test2 / len(testloader.dataset)
    testloss_res2[epoch] = _loss_test2
    _acc_test2 = acc_test2 / len(testloader.dataset)
    testacc_res2[epoch] = _acc_test2

    # print('Resnet1')
    # for param in resnet1.parameters():
    #     print(param)
    # print('\n\nResNet2')
    # for param in resnet2.parameters():
    #     print(param)

    print('Epoch {}\nResNet1    loss: {}, acc: {}\nResNet2    loss: {}, acc: {}'.format(epoch, _loss_test1, _acc_test1, _loss_test2, _acc_test2))



# np.save('D:/Dropbox/나메렝/wml/210610/ResNet18_ep100_loss.npy', testloss_res)
# np.save('D:/Dropbox/나메렝/wml/210610/ResNet18_ep100_acc.npy', testacc_res)

'''