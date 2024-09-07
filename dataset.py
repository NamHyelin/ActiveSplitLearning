import torch
import tensorflow as tf
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from model import num_classes
import numpy as np
import random
import copy
# from main_ct1 import num_clients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batchsize=128
num_datasplits=2.5 #10


def new_dataload(transform_normal=True):
    global trainset
    from torchvision import datasets
    if transform_normal==True:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif transform_normal==False:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)   #cifar는 permute
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batchsize)



transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader= DataLoader(testset, shuffle=True, batch_size= batchsize)


'''
y_train, seed= torch.sort(torch.tensor(trainset.targets))
x_train= torch.tensor(trainset.data[seed])




#TODO remove class (simpler classification problem)
target_indices_train= np.arange(len(trainset.targets))
target_indices_test= np.arange(len(testset.targets))

classidx_to_remove= np.array([0,1,2,3,4])

idx_to_keep_train= [i for i in range(len(trainset.targets)) if np.isin(trainset.targets[i], classidx_to_remove, invert=True).item()]
idx_to_keep_test= [i for i in range(len(testset.targets)) if np.isin(testset.targets[i], classidx_to_remove, invert=True).item()]
target_indices_train= target_indices_train[idx_to_keep_train]
target_indices_test= target_indices_test[idx_to_keep_test]

train_dataset= Subset(trainset, target_indices_train)
test_dataset= Subset(testset, target_indices_test)
train_dataset= SubsetRandomSampler(train_dataset)
test_dataset= SubsetRandomSampler(train_dataset)
train_loader= DataLoader(train_dataset, shuffle=False, batch_size=batchsize)
test_loader= DataLoader(test_dataset, shuffle=False, batch_size=batchsize)

'''

seed2= int(tf.random.uniform([1], 1, 10))
data_per_class= int(50000/num_classes)   #5000

#TODO split data for each client
class SplitData(Dataset):
    #train dataset 만 수행
    def __init__(self, client_idx, random_seed):
        self.client_idx= client_idx
        self.num_data_per_client = int(data_per_class / num_datasplits)
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # self.seed= int(tf.random.uniform([1], 1, 10))   # changed to seed2 -> label sync communication
        self.random_seed= random_seed

        # y_train, seed = torch.sort(torch.tensor(trainset.targets))   #trainset.targets  / train_dataset.dataset.targets
        # x_train = torch.tensor(trainset.data[seed])                  #trainset.data[seed]  / train_dataset.dataset.data[seed]

        np.random.seed(self.random_seed)
        np.random.shuffle(trainset.targets)
        np.random.seed(self.random_seed)
        np.random.shuffle(trainset.data)
        y_train, seed = torch.sort(torch.tensor(trainset.targets))
        x_train = torch.tensor(trainset.data[seed])


        x_cat = x_train[:data_per_class][(self.client_idx * self.num_data_per_client): ((self.client_idx + 1) * self.num_data_per_client)]
        for idx in range(num_classes-1):
            x_cat = torch.cat([x_cat, x_train[data_per_class * (idx + 1):data_per_class * (idx + 2)][(self.client_idx * self.num_data_per_client): ((self.client_idx + 1) * self.num_data_per_client)]])
        tf.random.set_seed(seed2)
        x_cat= tf.random.shuffle(x_cat)
        self.data= x_cat.numpy()

        y_cat = y_train[:data_per_class][(self.client_idx * self.num_data_per_client): ((self.client_idx + 1) * self.num_data_per_client)]
        for idx in range(num_classes-1):
            y_cat = torch.cat([y_cat, y_train[data_per_class * (idx + 1):data_per_class * (idx + 2)][(self.client_idx * self.num_data_per_client): ((self.client_idx + 1) * self.num_data_per_client)]])
        tf.random.set_seed(seed2)
        y_cat = tf.random.shuffle(y_cat)
        self.targets= y_cat.numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x= self.data[idx]
        x= self.transform(x)
        y= self.targets[idx]
        return x, y




# https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-3-ae2103c40c22?p=6c2e9494398b




def noniiddatasetsplit(num_users:int, devicesize : int, distribution:float ):
    class_num = num_classes
    np.random.seed(0)
    random.seed(0)

    states = [distribution for i in range(class_num)]
    t = np.random.dirichlet(states, num_users).transpose()
    s = t * devicesize
    devicedata=[]
    for num in range(num_users):
        devicedata.append([int(s.transpose()[num][i]) for i in range(class_num)])

    # class별 data수 통일
    for c in range(num_classes):
        summ=0
        for i in range(num_users):
            summ+=devicedata[i][c]
        if summ> (num_users*devicesize/class_num):
            remains=summ-int(num_users*devicesize/class_num)
            remains_divide= np.zeros((num_users))
            for i in range(num_users):
                if i==num_users-1:
                    remains_divide[i]=remains-np.sum(remains_divide)
                else:
                    remains_divide[i]=int(remains*(devicedata[i][c]/summ))
                devicedata[i][c] -= int(remains_divide[i])
        elif summ< (devicesize/class_num)*num_users:
            remains = int(num_users*devicesize / class_num) - summ
            remains_divide = np.zeros((num_users))
            for i in range(num_users):
                if i==num_users-1:
                    remains_divide[i]=remains-np.sum(remains_divide)
                else:
                    remains_divide[i]=int(remains*(devicedata[i][c]/summ))
                devicedata[i][c] += int(remains_divide[i])

    # client 별 data수 통일
    for i in range(num_users):
        if i!=num_users-1:
            summ= np.sum(devicedata[i])
            if summ>devicesize:
                remains=summ-devicesize
                arr = [0] * (num_classes);
                for r in range(remains):
                    arr[random.randint(0, remains) % (num_classes-1)] += 1
                devicedata[i]=np.subtract(devicedata[i],arr)
                devicedata[i+1] = np.add(devicedata[i+1], arr)
            elif summ<devicesize:
                remains=devicesize-summ
                arr = [0] * (num_classes);
                for r in range(remains):
                    arr[random.randint(0, remains) % (num_classes - 1)] += 1
                devicedata[i] = np.add(devicedata[i], arr)
                devicedata[i+1] = np.subtract(devicedata[i+1], arr)



    '''
    lack=[devicesize-(sum(devicedata[i])) for i in range(num_users)]
    for i in range(num_users):
        for j in range(lack[i]):
            x=devicedata[i].index(min(devicedata[i]))
            devicedata[i][x]=devicedata[i][x]+1
    '''

    return devicedata

def NoniidSplitData(trainset, batch_size, noniid_distribution, num_users, num_classes):
    idxs = [i for i in range(len(trainset))]
    # labels = dataset.train_labels.numpy()
    labels = np.array(trainset.targets)
    np.random.seed(0)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].tolist()
    indexes = []
    num_data_perclass= int(len(np.array(trainset.targets))/num_classes)
    for i in range(num_classes):
        indexes.append(idxs[num_data_perclass * i : num_data_perclass * (i + 1)])

    dataperdevice=noniiddatasetsplit(num_users, batch_size, noniid_distribution)
    # same sample numbers
    dict_users = {}
    for i in range(num_classes):
        for j in range(num_users):
            if i == 0:
                dict_users[j] = list(np.random.choice(indexes[i], dataperdevice[j][i], replace=False))  #원래 True인데 False도 적혀있음
                indexes[i] = list(set(indexes[i]) - set(dict_users[j]))
            else:
                dict_users[j] = dict_users[j] + list(np.random.choice(indexes[i], dataperdevice[j][i], replace=False)) #원래 True인데 False도 적혀있음
                indexes[i] = list(set(indexes[i]) - set(dict_users[j]))

    # dataset per device
    dataset_train = []

    for i in range(num_users):
        x = np.zeros(shape=(batch_size, 32, 32, 3), dtype=np.uint8)
        y = []
        for j in range(len(dict_users[i])):
            x[j, :, :, :] = trainset.data[dict_users[i][j], :, :, :]
            y.append(trainset.targets[dict_users[i][j]])
        dataset_train.append(copy.copy(trainset))
        dataset_train[i].data = x
        dataset_train[i].targets = y
        dataset_train[i].__setattr__('dict_users', dict_users)

    return dataset_train



# y_train= [y for _,y in client_1_trainset]
# counter_train= collections.Counter(y_train)
# print(counter_train)
