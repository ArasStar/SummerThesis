
import os
import torch




import relative_position
import jigsaw

head_libs={"Relative_Position":relative_position ,"Jigsaw":jigsaw}


class Load_Model(object):

    def __init__(self, method='', kwargs=None, from_checkpoint="", pre_trained ="", combo =[],
            optimizer=None, model=None, plot_loss=None, use_cuda=True):

        self.method = method
        self.from_checkpoint = from_checkpoint
        self.pre_trained = pre_trained
        self.param_names_dict = { "Common":["_num_epochs", "_batch_size", "_learning_rate"]  ,"Jigsaw":["_perm_set_size", "_grid_crop_size", "_patch_crop_size"],
                                "Relative_Position":["_split"]}
        self.combo = combo
        self.kwargs = kwargs
        self.optimizer= optimizer
        self.model= model
        self.head = None
        self.out_D ={"Relative_Position":8 ,"Jigsaw":self.kwargs["Jigsaw"]["perm_set_size"]}
        self.use_cuda = use_cuda

        if use_cuda and torch.cuda.is_available():
            #print("using CUDA")
            self.device = torch.device('cuda:0')
        else:
            print("CUDA didn't work")
            self.device = torch.device('cpu')

    def __call__(self):

        if self.from_checkpoint:
            self.load_from_checkpoint()
            file_name = self.get_file_name()

        elif self.pre_trained:
            pass

        else:
            file_name = self.get_file_name()
            self.set_head()

        #modelhead ,model, optimizer ,file_name, plot_loss
        return file_name, self.head

    def get_file_name(self):

        file_name = ""
        if self.method.__contains__("combination"):
            if self.combo ==[]:
                 print("PROBLEM file_name NAIVE COMB")
            file_name = self.method+"*"+"*".join(self.combo)+"*"

        else:
            file_name = self.method
            self.combo= [self.method]

        for c in ["Common"] + self.combo:
            args = self.kwargs[c]
            for param_name in self.param_names_dict[c]:
                param_idx = param_name[1:]
                param = args[param_idx]
                file_name = file_name + param_name + str(param)

        file_name = file_name + ".tar"

        return file_name

    def set_head(self):
        if self.combo ==[]: print("PROBLEM gethead NAIVE COMB")

        n_heads = len(self.combo)
        heads = []
        for h in self.combo:

          head_module = head_libs[h]
          method_head = head_module.Basic_Head(1024,self.out_D[h], gpu = self.use_cuda)
          to_patch = getattr(head_module,h)
          kwarg =self.kwargs[h]

          heads.append({"head":method_head, "patch_func":to_patch, "args": kwarg, "head_name":h})

        self.head = heads

    def load_from_checkpoint(self):

        from_checkpoint = self.from_checkpoint
        if self.from_checkpoint.__contains__("combination"):

          self.method = from_checkpoint[from_checkpoint.index("_supervised/")+12: from_checkpoint.index('combination')+11]
          self.combo = from_checkpoint[from_checkpoint.index("*")+1: from_checkpoint.index("*_num_epochs")].split("*")

        else:
          self.method = from_checkpoint[from_checkpoint.index("_supervised/")+12:from_checkpoint.index('_num_epochs')]
          self.combo = [method]

        #loading model
        checkpoint = torch.load(self.from_checkpoint, map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)#loading features
        self.plot_loss = checkpoint['loss']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in self.optimizer.state.values():
          for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=self.device)


        self.update_params_from_checkpoint()
        self.set_head()
        head_dict = checkpoint["model_head"] # dict headname: headstateDict
        for head in self.head:
            h_name = head["head_name"]
            head["head"].load_state_dict(head_dict[h_name])


    def update_params_from_checkpoint(self):

        from_checkpoint =self.from_checkpoint
        #aranging names and params
        #num_epochs
        start_i = from_checkpoint.index('_epochs')
        end_i =from_checkpoint.index('_batch')
        initial_epoch = int(from_checkpoint[start_i+7:end_i])
        self.kwargs["Common"]["num_epochs"]= initial_epoch +self.kwargs["Common"]["num_epochs"]
        '''
        #Batch
        start_i = from_checkpoint.index('batch_size')
        end_i =from_checkpoint.index('_learning_rate')
        batch_size = int(from_checkpoint[start_i+10:end_i])
        self.kwargs["Common"]["batch_size"]= batch
        '''
        if from_checkpoint.__contains__("split"):
            #split
            start_i = from_checkpoint.index('split')
            split = float(from_checkpoint[start_i+5: start_i+6])
            self.kwargs["Relative_Position"]["split"] = split

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

            old_perm = self.kwargs["Jigsaw"]["perm_set_size"]
            self.kwargs["Jigsaw"]["perm_set_size"] = perm_set_size
            self.kwargs["Jigsaw"]["grid_crop_size"] = grid_crop_size
            self.kwargs["Jigsaw"]["patch_crop_size"] = patch_crop_size

            self.kwargs["Jigsaw"]["path_permutation_set"].replace(str(old_perm)+".pt",str(perm_set_size)+".pt")