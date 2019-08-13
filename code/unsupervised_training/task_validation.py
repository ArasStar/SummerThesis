import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import f1_score,accuracy_score
import sys

saved_model_PATH="/vol/bitbucket/ay1218/"

root_PATH_dataset = "/vol/gpudata/ay1218/"
#root_PATH_dataset = saved_model_PATH


root_PATH = "/homes/ay1218/Desktop/"

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/chexpert_load')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/load_model')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/relative_position')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/rotation')

sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/selfsupervised_heads/jigsaw')
sys.path.insert(0, root_PATH+'SummerThesis/code/custom_lib/utilities/plotting_lib')

import chexpert_load
import relative_position
import jigsaw
import rotation

head_libs={"Relative_Position":relative_position ,"Jigsaw":jigsaw,"Rotation":rotation}

class Task_Validation(object):
    def __init__(self,full_file_path="", transform =  transforms.Compose([ transforms.Lambda(lambda x: torch.cat([x, x, x], 0))]) , gpu=True, resize=320,normalize=False):

        self.resize=resize
        self.gpu=gpu
        self.root_PATH = root_PATH
        self.saved_model_PATH = saved_model_PATH
        self.root_PATH_dataset = root_PATH_dataset
        self.file_name = full_file_path.split('/')[-1]
        self.file_path = full_file_path

        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.head_combo=[]
        self.kwargs= {"Relative_Position": {} ,"Jigsaw":{},"Rotation":{}}
        self.model = models.densenet121().to(self.device)
        self.out_D = {"Relative_Position":8 ,"Rotation":4,"Jigsaw":None}
        self.transform_after_patching =transform
        self.heads = None
        self.normalize= normalize

    def __call__(self):

        self.load_from_checkpoint()

        for head in self.heads:

            self.validate(head)


    def validate(self,head):

        if self.normalize:
            transform= transforms.Compose([  transforms.Resize((self.resize,self.resize)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

        else:
            transform= transforms.Compose([  transforms.Resize((self.resize,self.resize)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        cheXpert_valid_dataset, valid_dataloader = chexpert_load.chexpert_load(self.root_PATH+"SummerThesis/code/custom_lib/chexpert_load/valid.csv",transform,
                                                                 8, shuffle=False, root_dir = self.root_PATH_dataset)
        self.model.classifier = head["head"]
        method_name = head["head_name"]

        preds= []
        preds=np.array(preds)
        acts= []
        acts=np.array(acts)

        self.model.eval()
        with torch.no_grad():
            for images, _,_ in valid_dataloader:

                patcher = head["patch_func"](image_batch= images,**head["args"])
                patches, labels =  patcher()

                patches = patches.to(device = self.device, dtype = torch.float32)
                labels = labels.to(device=self.device, dtype=torch.long)

                outputs = self.model(patches)

                #sigmoid gets rounded
                predictions= torch.sigmoid(outputs).max(1)[1]# taking the max probabilty class
                #predictions=probs.round()

                preds = predictions.cpu().numpy()  if preds.size ==0 else np.concatenate((preds,predictions.cpu().numpy()))
                acts = labels.cpu().numpy()  if acts.size ==0 else np.concatenate((acts,labels.cpu().numpy()))

        f1_micro=f1_score(acts, preds, average="micro")
        f1_macro = f1_score(acts, preds, average="macro")
        acc = accuracy_score(acts,preds)

        path= "/".join(self.file_path.split("/")[:-1])+"/"

        file_obj = open(path+"/"+method_name+"_performance.txt","w")
        file_obj.write("f1_micro: "+str(f1_micro))
        file_obj.write("\n")
        file_obj.write("f1_macro: "+str(f1_macro))
        file_obj.write("\n")

        file_obj.write("accuracy: "+str(acc))
        file_obj.close()

    def load_from_checkpoint(self):

        file_name = self.file_name

        if self.file_name.__contains__("*"):
          self.head_combo = file_name[file_name.index("*")+1: file_name.index("*_num_epochs")].split("*")

        else:
          head_name = file_name[0:file_name.index('_num_epochs')]
          self.head_combo = [head_name]

        #loading model
        checkpoint = torch.load(self.file_path, map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)#loading features for model

        self.update_params_from_checkpoint()
        self.set_head()

        head_dict = checkpoint["model_head"] # dict headname: headstateDict

        for head in self.heads:

            h_name = head["head_name"]
            head["head"].load_state_dict(head_dict[h_name])


    def update_params_from_checkpoint(self):

        from_checkpoint =self.file_name

        if from_checkpoint.__contains__("patch_size"):
            #split
            start_i = from_checkpoint.index('patch_size')
            patch_size = int(from_checkpoint[start_i+10: start_i+12])
            self.kwargs["Relative_Position"]["patch_size"] = patch_size
            self.kwargs["Relative_Position"]["transform"] = self.transform_after_patching

        if from_checkpoint.__contains__("Rotation"):
            #split
            start_i = from_checkpoint.index('K')
            K = int(from_checkpoint[start_i+1: start_i+2])
            self.kwargs["Rotation"]["K"] = K
            self.kwargs["Rotation"]["transform"] = self.transform_after_patching

        if from_checkpoint.__contains__("perm_set"):
            #perm set size
            start_i = from_checkpoint.index('perm_set_size')
            end_i = from_checkpoint.index('_grid_crop_size')
            perm_set_size = int(from_checkpoint[start_i+13: end_i])
            #grid crop size
            start_i=end_i
            end_i = from_checkpoint.index('_patch_crop_size')
            grid_crop_size = int(from_checkpoint[start_i+15: end_i])
            #patch crop size
            start_i=end_i
            patch_crop_size = int(from_checkpoint[start_i+16: start_i+18])

            self.kwargs["Jigsaw"]["perm_set_size"] = perm_set_size
            self.kwargs["Jigsaw"]["grid_crop_size"] = grid_crop_size
            self.kwargs["Jigsaw"]["patch_crop_size"] = patch_crop_size

            self.kwargs["Jigsaw"]["path_permutation_set"] = self.root_PATH+"SummerThesis/code/custom_lib/utilities/permutation_set/saved_permutation_sets/permutation_set"+str(perm_set_size)+".pt"

            self.out_D["Jigsaw"] = perm_set_size
            self.kwargs["Jigsaw"]["transform"] = self.transform_after_patching


    def set_head(self):
        assert(self.head_combo !=[]),"PROBLEM file_name NAIVE COMB2"

        heads = []
        for h in self.head_combo:

            head_module = head_libs[h]
            method_head = head_module.Basic_Head(1024,self.out_D[h], gpu = self.gpu)
            to_patch = getattr(head_module,h)
            kwarg =self.kwargs[h]
            heads.append({"head":method_head, "patch_func":to_patch, "args": kwarg ,"head_name":h})

        self.heads = heads
