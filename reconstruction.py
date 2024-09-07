import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from dataset import batchsize
import matplotlib.pyplot as plt
from dataset import new_dataload, SplitData
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




client= torch.load('./client_model_interrupt_concat_bce.pth')
server= torch.load('./server_model_interrupt_concat_bce.pth')  #_reconstloss
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader= DataLoader(testset, shuffle=True, batch_size= 50)
new_dataload()
trainset = SplitData(0,1)  #split10
trainloader= DataLoader(trainset, shuffle=True, batch_size= 2)




return_layers_client = {   #앞에가 print 했을때 나오는 layer 이름  #key value
    'conv1': '1',
    'layer1.0.conv1' : 'layer1_1',
    'layer1.0.conv2' : 'layer1_2',
    'layer1.1.conv1' : 'layer1_3',
    'layer1.1.conv2' : 'layer1_4',
}


return_layers_server = {   #앞에가 print 했을때 나오는 layer 이름  #key value
    'layer2.0.conv1' : 'layer2_1',
    'layer2.0.conv2' : 'layer2_2',
    'layer2.1.conv1' : 'layer2_3',
    'layer2.1.conv2' : 'layer2_4',
    'layer2.0.conv1' : 'layer2_1',
    'layer2.0.conv2' : 'layer2_2',
    'layer2.1.conv1' : 'layer2_3',
    'layer2.1.conv2' : 'layer2_4',
    'layer3.0.conv1' : 'layer3_1',
    'layer3.0.conv2' : 'layer3_2',
    'layer3.1.conv1' : 'layer3_3',
    'layer3.1.conv2' : 'layer3_4',
    'layer4.0.conv1' : 'layer4_1',
    'layer4.0.conv2' : 'layer4_2',
    'layer4.1.conv1' : 'layer4_3',
    'layer4.1.conv2' : 'layer4_4'
}

input_sample= torch.stack([torch.tensor(testset.data[0]), torch.tensor(testset.data[1])], dim=0).permute(0,3,1,2).float().to(device)
mid_getter_client = MidGetter(client, return_layers=return_layers_client, keep_output=True)
mid_outputs_client, model_output_client = mid_getter_client(input_sample)

hidden_sample= client(input_sample)
mid_getter_server = MidGetter(server, return_layers=return_layers_server, keep_output=True)
mid_outputs_server, model_output_server = mid_getter_server(list(hidden_sample)[0])



class Decoder(nn.Module):
    def __init__(self, channel_size, feature_size):
        super(Decoder, self).__init__()
        self.channel_size=channel_size
        self.feature_size= feature_size

        if self.feature_size == 1:
            stride = 2
        elif self.feature_size == 2:
            stride = 2
        else:
            stride = 1

        if self.feature_size == 1:
            p1, p2 = 10, 12
        elif self.feature_size == 2:
            p1, p2 = 10, 12
        elif self.feature_size == 4:
            p1, p2 = 8, 8
        elif self.feature_size == 8:
            p1, p2 = 6, 8  # 6,8
        elif self.feature_size == 16:
            p1, p2 = 5, 5

        self.decoder = nn.Sequential(
            nn.Conv2d(self.channel_size, int(12), kernel_size=3, stride=stride, padding=p1),
            nn.BatchNorm2d(int(12)),
            nn.ReLU(),
            nn.Conv2d(int(12), 3, kernel_size=3, stride=1, padding=p2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


# for i in range(len(list(return_layers.keys()))):
#     print('1: ', mid_outputs[list(return_layers.values())[i]].shape)
#     decoder= Decoder(mid_outputs[list(return_layers.values())[i]].shape[1], mid_outputs[list(return_layers.values())[i]].shape[-1]).to(device)
#     y= decoder(mid_outputs[list(return_layers.values())[i]].to(device))
#     print('2: ',y.shape)


mutual_info_client = []
mutual_info_server = []

for i in range(len(list(return_layers_client.keys()))):
    hidden_size= mid_outputs_client[list(return_layers_client.values())[i]][0].shape
    decoder = Decoder(mid_outputs_client[list(return_layers_client.values())[i]].shape[1],
                      mid_outputs_client[list(return_layers_client.values())[i]].shape[-1]).to(device)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=0.001)
    # criterion= nn.BCELoss()
    criterion= nn.MSELoss()
    print('\n{}: '.format(i))
    for epoch in range(3):
        loss_=0.0
        for batch_idx, (input, label) in enumerate(trainloader):
            input, label = input.to(device), label.to(device)
            optimizer_d.zero_grad()
            mid_outputs, model_output = mid_getter_client(input)
            decoder_out= decoder(mid_outputs[list(return_layers_client.values())[i]].to(device))
            loss=criterion(decoder_out, input)
            loss_+=loss.item()
            loss.backward()
            optimizer_d.step()
        loss__=loss_/len(trainloader.dataset)
        if epoch==2:
            mutual_info_client.append(loss__)
        print('epoch{}: {}'.format(epoch, loss__))
np.save('D:/Dropbox/나메렝/wml/210806/mseloss_client_interrupt_concat_bce.npy', mutual_info_client)

#len(list(return_layers_server.keys()))
for i in range(len(list(return_layers_server.keys()))):
    hidden_size= mid_outputs_server[list(return_layers_server.values())[i]][0].shape
    decoder = Decoder(mid_outputs_server[list(return_layers_server.values())[i]].shape[1],
                      mid_outputs_server[list(return_layers_server.values())[i]].shape[-1]).to(device)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=0.001)
    # criterion= nn.BCELoss()
    criterion= nn.MSELoss()
    print('\n{}: '.format(i))
    for epoch in range(3):
        loss_=0.0
        for batch_idx, (input, label) in enumerate(trainloader):
            input, label = input.to(device), label.to(device)
            optimizer_d.zero_grad()
            hidden= client(input)
            mid_outputs, model_output = mid_getter_server(list(hidden)[0])
            decoder_out= decoder(mid_outputs[list(return_layers_server.values())[i]].to(device))
            loss=criterion(decoder_out, input)
            loss_+=loss.item()
            loss.backward()
            optimizer_d.step()
        loss__=loss_/len(trainloader.dataset)
        if epoch==2:
            mutual_info_server.append(loss__)
        print('epoch{}: {}'.format(epoch, loss__))
    np.save('D:/Dropbox/나메렝/wml/210806/mseloss_server_interrupt_concat_bce.npy', mutual_info_server) #_reconst


#loss니까 작을수록 mutual info가 큰거임 앞에서 작다가 점점 커져야해
#split10#split10#split10#split10#split10#split10#split10#split10#split10#split10


# plt.plot((mutual_info_client+mutual_info_server), '.-')
# plt.show()