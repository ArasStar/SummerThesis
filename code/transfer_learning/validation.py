import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import plot_loss_auc_n_precision_recall
from torchvision import transforms




class Validation(object):
    def __init__(self,bs=0,chexpert_load="", model="", plot_loss="",file_name="", root_PATH="", saved_model_PATH="",root_PATH_dataset = "", gpu=True, transform = ""):

        self.model = model
        self.plot_loss = plot_loss
        self.bs = bs
        self.transform = transform if transform else transforms.Compose([ transforms.Resize((320,320)), transforms.ToTensor(),transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.root_PATH = root_PATH
        self.saved_model_PATH = saved_model_PATH
        self.root_PATH_dataset = root_PATH_dataset
        self.file_name = file_name
        self.chexpert_load=chexpert_load
        self.device = torch.device('cuda') if gpu else torch.device('cpu')

    def __call__(self):

        cheXpert_valid_dataset, valid_dataloader = self.chexpert_load.chexpert_load(self.root_PATH+"SummerThesis/code/custom_lib/chexpert_load/valid.csv",self.transform,
                                                                     self.bs, shuffle=False, root_dir = self.root_PATH_dataset)

        probs= []
        probs=np.array(probs)
        acts= []
        acts=np.array(acts)

        self.model.eval()
        with torch.no_grad():
            for images, labels in valid_dataloader:

                images = images.to(device = self.device)
                labels = labels.to(device = self.device)
                outputs = self.model(images)

                #sigmoid gets rounded
                probabilities= torch.sigmoid(outputs)
                #predictions=probs.round()

                probs = probabilities.cpu().numpy()  if probs.size ==0 else np.vstack((probs,probabilities.cpu().numpy()))
                acts = labels.cpu().numpy()  if acts.size ==0 else np.vstack((acts,labels.cpu().numpy()))

        # SAVING PLOTS and models
        curves =plot_loss_auc_n_precision_recall.Curves_AUC_PrecionnRecall(self.chexpert_load,cheXpert_valid_dataset, probs, acts, model_name= self.file_name,root_PATH= self.saved_model_PATH)
        curves()
        curves.plot_loss(self.plot_loss)
        curves.auc_difference_print()
        print("HULOOOOOOOOOOOOOOOOOOOO")





#ola senor
