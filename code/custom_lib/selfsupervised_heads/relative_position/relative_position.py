import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches4rectangle


class Relative_Position(object):
    def __init__(self, image_batch,transform=None, show=False,patch_size=64 ,labels_path="/home/aras/Desktop/"):
        self.bs, _, self.h, _ = image_batch.shape
        self.init_h = self.h
        self.patch_size = patch_size
        self.jitter = 7
        self.patch_gap= 1.5*self.patch_size if patch_size<65 else 1.25*self.patch_size
        self.cropsize = self.patch_size + self.patch_gap*2 + self.jitter*2

        self.aug_size =352
        self.resize = torch.nn.Upsample(size = (self.aug_size,self.aug_size))
        self.h = self.aug_size if patch_size >64 else self.h

        self.show = show  # print cropped images and show where they are 1 time for each batch
        self.image_batch = image_batch
        self.transform = transform
        self.labels_path = labels_path+"SummerThesis/code/custom_lib/selfsupervised_heads/relative_position/rel_pos_labels.png"
        # list of [patch_xy,pathc2_xy,dir1 , dir reverse]
        self.adjacent_patch_comb = [[(0,0),(1,1),3.0],[(1,1),(2,2),3.0],[(0,1),(1,2),3.0],[(1,0),(2,1),3.0],
                                    [(0,2),(1,1),5.0],[(1,1),(2,0),5.0],[(0,1),(1,0),5.0],[(1,2),(2,1),5.0],
                                    [(0,0),(0,1),2.0],[(0,1),(0,2),2.0],[(1,0),(1,1),2.0],[(1,1),(1,2),2.0],[(2,0),(2,1),2.0],[(2,1),(2,2),2.0],
                                    [(0,0),(1,0),4.0],[(1,0),(2,0),4.0],[(0,1),(1,1),4.0],[(1,1),(2,1),4.0],[(0,2),(1,2),4.0],[(1,2),(2,2),4.0]]

    def __call__(self):

        patches = torch.Tensor()
        labels = np.empty(self.bs)

        #transform_patch = transforms.Compose([transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        for idx, image in enumerate(self.image_batch):

            if self.patch_size == 96:
                image = self.resize(image.view(1,3,self.init_h,-1)).view(3,self.h,self.h)

            [srow, scol] = np.random.randint(self.h - self.cropsize - 1, size=2)
            patch1, patch2, direction, cord_list1, cord_list2 , patch_cord = self.random_adjecent_patches(image, srow, scol)

            if (idx == 2 or idx == 0 ) and self.show:
                self.show_cropped_patches(image, patch1, patch2, direction, cord_list1, cord_list2, srow, scol, patch_cord)

            if self.transform:
                h_patch, w_patch = patch1.shape

                #patch1 = patch1.view(1, h_patch, w_patch)
                #patch2 = patch2.view(1, h_patch, w_patch)

                patch1 = self.transform(patch1)
                patch2 = self.transform(patch2)

                #patch1 = patch1.view(1, 3, h_patch, w_patch)
                #patch2 = patch2.view(1, 3, h_patch, w_patch)
            else:
                patch1 = patch1.view(1, 3, self.patch_size, self.patch_size)
                patch2 = patch2.view(1, 3, self.patch_size, self.patch_size)

            pair = torch.cat((patch1, patch2))

            patches = torch.cat((patches, pair))
            #print("patches",patches.shape)

            labels[idx] = direction

        labels = torch.from_numpy(labels)
        return patches, labels


    def get_jitter(self):
        return np.random.choice([1.0,-1.0])*np.random.randint(self.jitter+1) + self.jitter

    def random_adjecent_patches(self, image, srow, scol):

        patch_cord = np.random.randint(len(self.adjacent_patch_comb))#len is 20 , picks random adjacent pair
        patch_cord =self.adjacent_patch_comb[patch_cord]

        patch1, patch2, direction, cord_list1, cord_list2 = self.patches_from_grid(patch_cord, image, srow, scol)

        return patch1, patch2, direction, cord_list1, cord_list2 , patch_cord

    def patches_from_grid(self, patch_cord, image, srow, scol):


        random_order = np.random.randint(2)

        a_row_jitter = self.get_jitter()
        a_col_jitter = self.get_jitter()
        b_row_jitter = self.get_jitter()
        b_col_jitter = self.get_jitter()

        a_row, a_col = int(srow + a_row_jitter + patch_cord[0][0]*self.patch_gap),  int(scol + a_col_jitter + patch_cord[0][1]*self.patch_gap)
        b_row, b_col = int(srow + b_row_jitter + patch_cord[1][0]*self.patch_gap),  int(scol + b_col_jitter + patch_cord[1][1]*self.patch_gap)

        patch_a = image[:, a_row:a_row+self.patch_size , a_col:a_col+self.patch_size]
        patch_b = image[:, b_row:b_row+self.patch_size, b_col:b_col+self.patch_size]
        dir = patch_cord[-1]


        if random_order:
            return patch_a, patch_b, dir ,[(a_row,a_row+self.patch_size),(a_col,a_col+self.patch_size)],[(b_row,b_row+self.patch_size),(b_col,b_col+self.patch_size)]
        else:
            dir = (dir + 4) % 8
            return patch_b, patch_a, dir, [(b_row,b_row+self.patch_size),(b_col,b_col+self.patch_size)] ,[(a_row,a_row+self.patch_size),(a_col,a_col+self.patch_size)]

    def show_cropped_patches(self, image, patch1, patch2, direction, cord_list1, cord_list2, srow, scol, patch_cord):
        # cordlist->[(row_s,row_e),(col_s,col_e)]
        gridimage_delete_later = image.clone()
        gridimage_delete_later = gridimage_delete_later.view(self.h, self.h)

        label_img = mpimg.imread(self.labels_path)
        plt.imshow(label_img)

        # Ploting
        print("cordlist->[(row_s,row_e),(col_s,col_e)]")
        print("cordlist1", cord_list1)
        print("cordlist2", cord_list2)

        row_start1 = cord_list1[0][0]
        col_start1 = cord_list1[1][0]

        row_start2 = cord_list2[0][0]
        col_start2 = cord_list2[1][0]

        fig, ax = plt.subplots(1)

        pil_gridimage_delete_later = transforms.ToPILImage()(gridimage_delete_later[0])
        ax.imshow(pil_gridimage_delete_later, cmap='Greys_r')
        rect = patches4rectangle.Rectangle((col_start1, row_start1), self.patch_size-2, self.patch_size-2, linewidth=3,
                                           edgecolor='r', facecolor='none',ls ="--")
        ax.add_patch(rect)
        rect = patches4rectangle.Rectangle((col_start2, row_start2), self.patch_size - 2, self.patch_size - 2, linewidth=3,
                                           edgecolor='b', facecolor='none',ls ="--")
        ax.add_patch(rect)

        for col in range(3):
            for row in range(3):

                if (row,col)!= patch_cord[0] and (row,col)!= patch_cord[1] :
                    row_jitter = self.get_jitter()
                    col_jitter = self.get_jitter()

                    row_start, col_start = int(srow + row_jitter + row*self.patch_gap),  int(scol + col_jitter + col*self.patch_gap)

                    rect = patches4rectangle.Rectangle((col_start, row_start), self.patch_size - 2, self.patch_size - 2, linewidth=1,
                                                   edgecolor='yellow', facecolor='none',ls ="--")
                    ax.add_patch(rect)


        plt.title('fullimage - label=' + str(direction))

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        pil_patch1 = transforms.ToPILImage()(patch1[0])
        plt.imshow(pil_patch1, cmap='Greys_r')
        plt.title('patch1-red',color='red')

        plt.subplot(1, 2, 2)
        pil_patch2 = transforms.ToPILImage()(patch2[0])
        plt.imshow(pil_patch2, cmap='Greys_r')
        plt.title('patch2-blue',color='blue')
        plt.show()



class Basic_Head(torch.nn.Module):
    def __init__(self, D_in, D_out=8, gpu=True):
        """
        No task head just concating what comes out just before default classifer then applying linear
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Basic_Head, self).__init__()
        self.device = torch.device('cuda:0') if gpu else torch.device('cpu')
        self.classifier = torch.nn.Linear(D_in*2,D_out).to(device = self.device)

    def forward(self, x):

        N,_ = x.shape
        # combining two representation (batch is [bs,ch,h,w ])
        x = torch.stack([torch.cat((x[idx,:],x[idx+1,:]))
                     for idx in range(0,N,2)], dim=0).to(device = self.device)

        #linear output with 8 outpts(directions)
        y_pred = self.classifier(x)
        return y_pred
