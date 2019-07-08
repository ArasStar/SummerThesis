import os#for CUDA tracking
import torch
import pandas as pd
from skimage.io import imread
#from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#MODELS
import re
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

chexpert_auc_scores = {'Atelectasis':      0.858,
                           'Cardiomegaly':     0.854,
                           'Consolidation':    0.939,
                           'Edema':            0.941,
                           'Pleural Effusion': 0.936}


radiologist_auc_Atelectasis= {'FPR': [0.21,0.18,0.31,0.22], 'TPR':[0.80,0.71,0.92,0.89]}

radiologist_auc_Cardiomegaly= {'FPR': [0.05,0.23,0.11,0.08], 'TPR':  [0.48,0.85,0.70,0.75]}

radiologist_auc_Consolidation= {'FPR':[0.11,0.09,0.03,0.05] , 'TPR': [0.66,0.48,0.45,0.52]}

radiologist_auc_Edema= {'FPR': [0.09,0.19,0.07,0.08], 'TPR': [0.63,0.79,0.58,0.68]}

radiologist_auc_Pleural_Effusion= {'FPR': [0.05,0.17,0.14,0.10], 'TPR': [0.82,0.83,0.89,0.89]}

radiologist_auc={'Atelectasis': radiologist_auc_Atelectasis,
                           'Cardiomegaly':     radiologist_auc_Cardiomegaly,
                           'Consolidation':    radiologist_auc_Consolidation,
                           'Edema':            radiologist_auc_Edema,
                           'Pleural Effusion': radiologist_auc_Pleural_Effusion}



radiologist_precision_recall_Atelectasis= {"Sensitivity": [0.80,0.71,0.92,0.89], "Precision": [0.62,0.64,0.56,0.64]}

radiologist_precision_recall_Cardiomegaly= {'Sensitivity': [0.48,0.85,0.70,0.75], "Precision": [0.82,0.61,0.74,0.80]}

radiologist_precision_recall_Consolidation= {"Sensitivity": [0.66,0.48,0.45,0.52] ,"Precision": [0.27,0.25,0.45,0.38]}


radiologist_precision_recall_Edema= {"Sensitivity": [0.63,0.79,0.58,0.68] ,"Precision": [0.58,0.44,0.59,0.62]}

radiologist_precision_recall_Pleural_Effusion= {"Sensitivity" :[0.82,0.83,0.89,0.89] ,"Precision" :[0.80,0.55,0.63,0.71]}

radiologist_precision_recall={'Atelectasis': radiologist_precision_recall_Atelectasis,
                           'Cardiomegaly':     radiologist_precision_recall_Cardiomegaly,
                           'Consolidation':    radiologist_precision_recall_Consolidation,
                           'Edema':            radiologist_precision_recall_Edema,
                           'Pleural Effusion': radiologist_precision_recall_Pleural_Effusion}




def load_posw(list_classes =  ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'] ):
    posw = {'Atelectasis': 2.3288235664367676, 'Cardiomegaly': 7.274592399597168,'Consolidation': 14.112899780273438, 'Edema': 2.4250192642211914, 'Pleural Effusion': 1.5922006368637085}
    posw_list=[posw[x]for x in list_classes]

    return torch.tensor(posw_list)


class CheXpertDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, labels_path=None, list_classes=None ,path=""):

        #super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.list_classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        self.observations_frame = pd.read_csv(csv_file)
        self.observations_frame['patient'] = self.observations_frame.Path.str.split('/',3,True)[2]
        self.observations_frame['study'] = self.observations_frame.Path.str.split('/',4,True)[3]

        #full_df['feature_string'] = full_df.apply(feature_string, axis = 1).fillna('')

        if labels_path:
            self.labels = torch.load(labels_path).type(torch.FloatTensor)

            if list_classes:
                list_idx = [ np.where(np.array(self.list_classes) == feature)[0][0] for feature in list_classes]
                self.labels = self.labels.index_select(1,torch.tensor(list_idx))

            self.list_classes=list_classes

        elif csv_file.__contains__("valid"):
            self.labels = torch.from_numpy(observations_frame.loc[:, self.list_classes].values).type(torch.FloatTensor)


        else:
            self.list_classes = list_classes if list_classes else self.list_classes
            u_one_features = ['Atelectasis', 'Edema']; u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
            u_one_features = [value for value in u_one_features if value in self.list_classes]
            u_zero_features = [value for value in u_zero_features if value in self.list_classes]

            labels = torch.Tensor()
            for _,row in self.observations_frame.iterrows():
                np_arr = self.feature_string(row,u_one_features,u_zero_features)
                tensor_arr = torch.from_numpy(np_arr).type(torch.FloatTensor).view(1,-1)
                labels = torch.cat((labels,tensor_arr),dim=0)

            self.labels = labels
            #torch.save(self.labels,"/home/aras/Desktop/SummerThesis/code/custom_lib/chexpert_load/self_train_labels.pt")

       # df_labels = pd.DataFrame(labels.numpy(),columns=[cl+"_feat" for cl in self.list_classes])
       #self.observations_frame.join(df_labels)


    def __len__(self):
        return len(self.observations_frame)

    def __getitem__(self, idx):

        #img_name = os.path.join(os.getcwd(),self.root_dir,self.observations_frame.iloc[idx, 0])
        img_name =  self.root_dir + self.observations_frame.loc[idx, ['Path']].values[0]
        #print("hoooop",img_name)
        image = imread(img_name)
        image = transforms.ToPILImage()(image)

        labels = self.labels[idx,:]

        # RETURNING IMAGE AND LABEL SEPERATELY FOR TORCH TRANSFORM LIB
        if self.transform:
            image = self.transform(image)

        return image, labels

    def feature_string(self,row, u_one_features, u_zero_features):

        feature_list = np.ones(len(self.list_classes))

        for feature in u_one_features:

            idx =np.where(np.array(self.list_classes) == feature)[0][0]
            if row[feature] in [-1,1]:
                feature_list[idx]=1
            else:
                feature_list[idx]=0

        for feature in u_zero_features:
            idx =np.where(np.array(self.list_classes) == feature)[0][0]

            if row[feature] == 1:
                feature_list[idx]=1
            else:
                feature_list[idx]=0

        return feature_list

def chexpert_load(csv_file_name, transformation, batch_size,labels_path=None,list_classes=None,shuffle=True,root_dir=""):

    cheXpert_dataset = CheXpertDataset(csv_file=csv_file_name,
                                       root_dir=root_dir, transform=transformation,labels_path=labels_path,list_classes=list_classes)

    dataloader = DataLoader(cheXpert_dataset, batch_size=batch_size, shuffle=shuffle)

    return cheXpert_dataset, dataloader
