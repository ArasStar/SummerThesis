import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches4rectangle


class Jigsaw(object):
    def __init__(self, image_batch,perm_set_size, path_permutation_set, grid_crop_size=255, patch_crop_size=64, gpu=True, show=False,transform= None):
        print("gpu",gpu)
        self.bs, _, self.h, self.w = image_batch.shape
        self.show = show  # print cropped images and show where they are 1 time for each batch
        self.image_batch = image_batch

        self.grid_crop_size = grid_crop_size
        assert (self.grid_crop_size %3 == 0), "has grid crop size has to be divisable by 3"
        self.grid_patch_size = int(self.grid_crop_size / 3)
        self.patch_crop_size = patch_crop_size
        assert (self.grid_patch_size >= self.patch_crop_size), "patch crop size has to be smaller then grid patch size"

        self.device = torch.device('cuda') if gpu else torch.device('cpu')
        assert(path_permutation_set.__contains__(str(perm_set_size))), "path file and perm set size not matching"
        self.permutation_set = torch.load(path_permutation_set, map_location=self.device)
        self.transform = transform

    def __call__(self):

        patches = torch.Tensor()
        labels = np.empty(self.bs)

        for idx, image in enumerate(self.image_batch):

            permuted_patches, label, shiftcol, shiftrow, cord_patches  = self.puzzled_patches(image)

            if idx == 1 and self.show:
                self.show_cropped_patches(image, shiftcol, shiftrow, permuted_patches, cord_patches, label)

            if self.transform:
                 permuted_patches = torch.stack([self.transform(x_i)
                     for i, x_i in enumerate(torch.unbind(permuted_patches, dim=0))], dim=0)

            patches = torch.cat((patches, permuted_patches))
            labels[idx] = label

        labels = torch.from_numpy(labels).type(torch.FloatTensor)
        return patches, labels

    def puzzled_patches(self, image):

        grid, shiftcol, shiftrow = self.random_crop(image, self.grid_crop_size)
        patches, cord_patches = self.get_patches(grid)
        permuted_patches, label = self.permut(patches)

        return permuted_patches, label, shiftcol, shiftrow, cord_patches

    def random_crop(self, image, cropsize):
        _, h, _ = image.shape
        [shiftrow, shiftcol] = np.random.randint(h - cropsize - 1, size=2)

        return image[:, shiftrow:shiftrow + cropsize, shiftcol:shiftcol + cropsize], shiftcol , shiftrow

    def get_patches(self, grid):

        # dive a 225x225 grid to 3x3
        patches = torch.empty(9,self.patch_crop_size, self.patch_crop_size)
        #non_transformed_patches = torch.empty(9,self.patch_crop_size, self.patch_crop_size)

        cord_patches = []
        patch_n = 0
        for i in range(0, 3):
            for j in range(0, 3):
                # each patch take random crop to prevent edge following

                row_start = 0 + self.grid_patch_size * i;
                row_finit = row_start + self.grid_patch_size
                col_start = 0 + self.grid_patch_size * j;
                col_finit = col_start + self.grid_patch_size

                if self.grid_patch_size > self.patch_crop_size:

                    patches[patch_n,:, :], scol, srow = self.random_crop(grid[:, row_start:row_finit, col_start:col_finit],
                                                                      self.patch_crop_size)
                    cord_patches.append((scol, srow))
                else:
                    patches[patch_n,:, :] = grid[:, row_start:row_finit, col_start:col_finit]
                    cord_patches.append((0, 0))

                patch_n = patch_n + 1

        return patches, cord_patches

    def permut(self, patches):
        set_size, _ = self.permutation_set.shape  # Nx9

        pick = np.random.randint(set_size)  # pick = label of the perm
        perm = self.permutation_set[pick]

        permuted_patches = patches[perm.long()]

        return permuted_patches, pick

    def show_cropped_patches(self, image, shift_col, shift_row, permuted_patches, cord_patches, label):
        # Preparing

        image_draw = transforms.ToPILImage()(image)
        perm_set = self.permutation_set[label]
        perm_label =(perm_set.cpu().numpy())
        # font = ImageFont.truetype("arial.ttf", fontsize)
        # Original Image plot
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.axis("off")

        start_col = shift_col
        start_row = shift_row
        for i in range(0, 4):
            shift = i * self.grid_patch_size
            ax.plot([start_col, start_col + self.grid_crop_size], [start_row + shift, start_row + shift], color='r',
                    linestyle='dashed')
            ax.plot([start_col + shift, start_col + shift], [start_row, start_row + self.grid_crop_size], color='r',
                    linestyle='dashed')

        fig.suptitle("grid size "+str(self.grid_crop_size)+", patch_size(green)"+str(self.patch_crop_size))

        # print(cord_patcstart_colstart_colhes)
        for idx, (srow, scol) in enumerate(cord_patches):
            # print(idx)label
            n = perm_set[idx]

            if n < 3:
                i = 0
            elif n < 6:
                i = 1
            else:
                i = 2

            j = n % 3

            rect = patches4rectangle.Rectangle((shift_col + j * self.grid_patch_size + scol, shift_row + i * self.grid_patch_size + srow),
                                                                                self.patch_crop_size, self.patch_crop_size , linewidth=2,edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        plt.imshow(image_draw, cmap='Greys_r')

        # Permuted Grid
        fig2, ax2 = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(5, 5))
        fig2.subplots_adjust(hspace=0, wspace=0)

        for idx in np.arange(9):

            if idx < 3:
                i = 0
            elif idx < 6:
                i = 1
            else:
                i = 2

            j = idx % 3

            patch_show = transforms.ToPILImage()(permuted_patches[idx,:, :])
            ax2[i, j].imshow(patch_show, cmap='Greys_r')
            ax2[i, j].axis('off')

        fig2.suptitle(str(perm_label))

        # Aligned Grid
        fig3, ax3 = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(5, 5))
        fig3.subplots_adjust(hspace=0, wspace=0)

        for index, idx in enumerate(perm_set):

            idx = int(idx.item())

            if idx < 3:
                i = 0
            elif idx < 6:
                i = 1
            else:
                i = 2

            j = idx % 3

            patch_show = transforms.ToPILImage()(permuted_patches[index, :, :])
            ax3[i, j].imshow(patch_show, cmap='Greys_r')
            ax3[i, j].axis('off')


        fig3.suptitle(str(np.arange(9)))

        plt.show()


class Basic_Head(torch.nn.Module):
    def __init__(self, D_in, D_out, gpu=True):
        """
        No task head just concating what comes out just before default classifer then applying linear
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Basic_Head, self).__init__()
        self.device = torch.device('cuda:0') if gpu else torch.device('cpu')
        self.classifier = torch.nn.Linear(D_in*9,D_out).to(device = self.device)


    def forward(self, x):
        N,_ = x.shape
        # combining two representation (batch is [bs,ch,h,w])

        def stack_tiles(tiles):
          tile_stacked = torch.Tensor().to(device = self.device)
          for tile in tiles:
            tile_stacked = torch.cat((tile_stacked,tile)).to(device = self.device)
          return tile_stacked

        x = torch.stack([stack_tiles(x[idx:idx+9,:].to(device=self.device)) for idx in range(0,N,9)], dim=0).to(device=self.device)

        #linear output with 8 outpts(directions)
        y_pred = self.classifier(x)
        return y_pred
