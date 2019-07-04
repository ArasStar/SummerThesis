class Basic_JigsawHead(torch.nn.Module):
    def __init__(self, D_in, D_out, gpu=True):
        """
        No task head just concating what comes out just before default classifer then applying linear
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Basic_JigsawHead, self).__init__()
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.classifier = torch.nn.Linear(D_in*9,D_out).to(device = self.device)
        

    def forward(self, x):        
        N,_ = x.shape
        # combining two representation (batch is [bs,ch,h,w])
      
        def stack_tiles(tiles):
          tile_stacked = torch.Tensor().to(device = self.device)
          for tile in tiles:
            tile_stacked = torch.cat((tile_stacked,tile)).to(device = self.device)
          return tile_stacked
          
        x = torch.stack([stack_tiles(x[idx:idx+9,:]) for idx in range(0,N,9)], dim=0).to(device=self.device)
        
        #linear output with 8 outpts(directions)
        y_pred = self.classifier(x)
        return y_pred


class Basic_RelativePositionHead(torch.nn.Module):
    def __init__(self, D_in, D_out=8, gpu=True):
        """
        No task head just concating what comes out just before default classifer then applying linear
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Basic_RelativePositionHead, self).__init__()
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.classifier = torch.nn.Linear(D_in*2,D_out).to(device = self.device)

    def forward(self, x):
                
        N,_ = x.shape
        # combining two representation (batch is [bs,ch,h,w ])
        x = torch.stack([torch.cat((x[idx,:],x[idx+1,:]))
                     for idx in range(0,N,2)], dim=0).to(device = self.device)
        
        #linear output with 8 outpts(directions)
        y_pred = self.classifier(x)
        return y_pred
    
class Naive_Combiation_Head(torch.nn.Module):
    def __init__(self, D_in, D_out=8, gpu=True):
        """
        No task head just concating what comes out just before default classifer then applying linear
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Basic_RelativePositionHead, self).__init__()
        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        self.classifier = torch.nn.Linear(D_in*2,D_out).to(device = self.device)

    def forward(self, x):
                
        N,_ = x.shape
        # combining two representation (batch is [bs,ch,h,w ])
        x = torch.stack([torch.cat((x[idx,:],x[idx+1,:]))
                     for idx in range(0,N,2)], dim=0).to(device = self.device)
        
        #linear output with 8 outpts(directions)
        y_pred = self.classifier(x)
        return y_pred
