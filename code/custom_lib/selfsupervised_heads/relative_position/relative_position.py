import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches4rectangle


class Relative_Position(object):
    def __init__(self, image_batch, split,transform=None, show=False,patch_size=64 ,labels_path="/home/aras/Desktop/"):
        self.bs, _, self.h, self.w = image_batch.shape
        self.cropsize = 266 if patch_size == 64 else 398 # else if patch size 96
        self.jitter = 5 if  patch_size == 64 else 7
        self.patch_size = patch_size

        self.split = split  # 3x3 or 2x2 grid
        self.i_row = round(self.h / self.split)
        self.i_col = round(self.w / self.split)
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
            patch1, patch2, direction, cord_list1, cord_list2 = self.random_adjecent_patches(image)

            if (idx == 2 or idx == 0 ) and self.show:
                self.show_cropped_patches(image, patch1, patch2, direction, cord_list1, cord_list2)

            if self.transform:
                patch1 = self.transform(patch1)
                patch2 = self.transform(patch2)
                _, h_patch, w_patch = patch1.shape
                patch1 = patch1.view(1, 3, h_patch, w_patch)
                patch2 = patch2.view(1, 3, h_patch, w_patch)
            else:
                _, h_patch, w_patch = patch1.shape
                patch1 = patch1.view(1, 1, h_patch, w_patch)
                patch2 = patch2.view(1, 1, h_patch, w_patch)
            pair = torch.cat((patch1, patch2))

            patches = torch.cat((patches, pair))
            #print("patches",patches.shape)

            labels[idx] = direction

        labels = torch.from_numpy(labels)
        return patches, labels


    def jitter(self):
        return np.random.choice([1,-1])*np.randint(self.jitter+1)

    def random_adjecent_patches(self, image):

        patch_cord = np.random.randint(len(self.adjacent_patch_comb))#len is 20 , picks random adjacent pair

        [shiftrow, shiftcol] = np.random.randint(self.h - self.cropsize - 1, size=2)

        patch1, patch2, direction cord_list1, cordlist2 = self.patches_from_grid(patch_cord, image, shiftcol, shiftrow)

        return patch1, patch2, direction, cord_list1, cord_list2

    def patches_from_grid(self, grid, patch_cord, srow ,scol):

        patch_gap = int(self.patchsize*1.5)

        srow += no_j_srow
        scol += no_j_scol

        random_order = np.random.randint(2)

        a_row_jitter = self.jitter()
        a_col_jitter = self.jitter()
        b_row_jitter = self.jitter()
        b_col_jitter = self.jitter()

        a_row, a_col = srow + a_row_jitter + patch_cord[0][0]*patch_gap,  scol + a_col_jitter + patch_cord[0][1]*patch_gap
        b_row, b_col = srow + b_row_jitter + patch_cord[1][1]*patch_gap,  scol + b_col_jitter + patch_cord[1][1]*patch_gap

        patch_a = grid[a_row:a_row+self.patch_size + , a_col:a_col+self.patch_size]
        patch_b = grid[b_row:b_row+self.patch_size, b_col:b_col+self.patch_size]
        dir = patch_cord[-1]


        if random_order:
            return patch_a, patch_b, dir ,[(a_row,a_row+self.patch_size),(a_col,a_col+self.patch_size)],[(b_row,b_row+self.patch_size),(b_col,b_col+self.patch_size)]
        else:
            dir = (dir + 4) % 8

            return patch_b, patch_a, dir, [(b_row,b_row+self.patch_size),(b_col,b_col+self.patch_size)] ,[(a_row,a_row+self.patch_size),(a_col,a_col+self.patch_size)]

        pass

    def patch_from_image(self, image, drow, dcol):

        i_row = int(self.i_row)
        i_col = int(self.i_col)
        max = self.split - 1

        patch = torch.zeros(1, i_row + 1, i_col + 1)

        row_s = i_row * drow
        col_s = i_col * dcol
        row_e = (row_s + i_row) if drow < max else self.h
        col_e = (col_s + i_col) if dcol < max else self.w

        patch[:, :row_e - row_s, :col_e - col_s] = image[:, row_s:row_e, col_s:col_e].clone()

        return patch, [(row_s, row_e), (col_s, col_e)]

    def show_cropped_patches(self, image, patch1, patch2, direction, cord_list1, cord_list2):
        # cordlist->[(row_s,row_e),(col_s,col_e)]
        gridimage_delete_later = image.clone()
        gridimage_delete_later = gridimage_delete_later.view(self.h, self.w)

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

        pil_gridimage_delete_later = transforms.ToPILImage()(gridimage_delete_later)
        ax.imshow(pil_gridimage_delete_later, cmap='Greys_r')
        rect = patches4rectangle.Rectangle((col_start1, row_start1), self.i_col - 2, self.i_row - 2, linewidth=3,
                                           edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        rect = patches4rectangle.Rectangle((col_start2, row_start2), self.i_col - 2, self.i_row - 2, linewidth=3,
                                           edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        plt.title('fullimage - label=' + str(direction))

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        pil_patch1 = transforms.ToPILImage()(patch1)
        plt.imshow(pil_patch1, cmap='Greys_r')
        plt.title('patch1-red',color='red')

        plt.subplot(1, 2, 2)
        pil_patch2 = transforms.ToPILImage()(patch2)
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
