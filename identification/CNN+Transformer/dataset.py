import numpy as np
import os
from torch.utils.data import DataLoader,Dataset

class PhoneDataset(Dataset):
    def __init__(self,path: str,train: bool) -> None:
        super().__init__()
        self.train = train
        self.path = path
        self.features,self.labels = self._load_data()
    
    def _load_data(self):
        pre_path=""
        if self.train:
            pre_path = "train"
        else :
            pre_path = "test"
        features = []
        feature_path = os.path.join(self.path,pre_path,"Inertial Signals")
        files = os.listdir(feature_path)
        files.sort(key=str.lower)
        #['train_acc_x.txt', 'train_acc_y.txt', 'train_acc_z.txt', 'train_gyr_x.txt', 'train_gyr_y.txt', 'train_gyr_z.txt']
        for feature_file in files:
            fileName = os.path.join(feature_path,feature_file)
            file = open(fileName, 'r')
            features.append(
                [np.array(cell, dtype=np.float32) for cell in [
                    row.strip().split(' ') for row in file
                ]]
            )
            file.close()
            #X_signals = 6*totalStepNum*128
        features = np.transpose(np.array(features), (1, 0, 2))#(totalStepNum*6*128)
        features = features.reshape(-1,1,6,128)#(Batch_size*1*6*128)B,C,H,W

        label_file_path = os.path.join(self.path,pre_path,f"y_{pre_path}.txt")
        file = open(label_file_path, 'r')
        # Read dataset from disk, dealing with text file's syntax
        labels = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],dtype=np.int32
        )
        file.close()
        # Substract 1 to each output class for friendly 0-based indexing
        labels = labels - 1
        # #one_hot
        # labels = labels.reshape(len(labels))
        # n_values = int(np.max(labels)) + 1
        # labels = np.eye(n_values)[np.array(labels, dtype=np.int32)]#(totalStepNum*118)

        return features,labels
    def __getitem__(self,index):
        return self.features[index],int(self.labels[index])

    def __len__(self):
        return len(self.features)


# if __name__ == "__main__":
#     dataloader = DataLoader(PhoneDataset("../data/Dataset#1",True),batch_size=512)
#     for i,(x,y) in enumerate(dataloader):
#         print(i,x.shape,y.shape)