import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches4rectangle


class Patch(object):
    def __init__(self, image_batch, split, show=False):

        self.bs, _, self.h, self.w = image_batch.shape
        self.split = split  # 3x3 or 2x2 grid
        self.i_row = round(self.h / self.split)
        self.i_col = round(self.w / self.split)
        self.show = show  # print cropped images and show where they are 1 time for each batch
        self.image_batch = image_batch

    def __call__(self):

        patches = torch.Tensor()
        labels = np.empty(self.bs)

        #transform_patch = transforms.Compose([transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        for idx, image in enumerate(self.image_batch):

            patch1, patch2, direction, cord_list1, cord_list2 = self.random_adjecent_patches(image)

            if idx == 2 and self.show:
                self.show_cropped_patches(image, patch1, patch2, direction, cord_list1, cord_list2)

            _, h_patch, w_patch = patch1.shape

            #patch1 = transform_patch(patch1).view(1, 3, h_patch, w_patch)
            #patch2 = transform_patch(patch2).view(1, 3, h_patch, w_patch)

            patch1 = patch1.view(1, 1, h_patch, w_patch)
            patch2 = patch2.view(1, 1, h_patch, w_patch)

            pair = torch.cat((patch1, patch2))

            #print("pair",pair.shape)
            patches = torch.cat((patches, pair))
            #print("patches",patches.shape)

            labels[idx] = direction

        return patches, labels

    def random_adjecent_patches(self, image):

        [drow, dcol] = np.random.randint(self.split, size=2)

        patch1, cord_list1 = self.patch_from_image(image, drow, dcol)
        patch2, direction, cord_list2 = self.pick_rand_adjacent_patch(image, drow, dcol)

        return patch1, patch2, direction, cord_list1, cord_list2

    def patch_from_image(self, image, drow, dcol):

        i_row = self.i_row
        i_col = self.i_col
        max = self.split - 1

        patch = torch.zeros(1, i_row + 1, i_col + 1)

        row_s = i_row * drow
        col_s = i_col * dcol
        row_e = (row_s + i_row) if drow < max else self.h
        col_e = (col_s + i_col) if dcol < max else self.w

        patch[:, :row_e - row_s, :col_e - col_s] = image[:, row_s:row_e, col_s:col_e].clone()

        return patch, [(row_s, row_e), (col_s, col_e)]

    def pick_rand_adjacent_patch(self, image, drow, dcol):

        def get_adjacent_indices(i, j):
            split = self.split
            adjacent_indices = []

            if i > 0:
                adjacent_indices.append((i - 1, j, 0.0))
                if j > 0:
                    adjacent_indices.append((i - 1, j - 1, 7.0))
                if j + 1 < split:
                    adjacent_indices.append((i - 1, j + 1, 1.0))

            if i + 1 < split:
                adjacent_indices.append((i + 1, j, 4.0))
                if j > 0:
                    adjacent_indices.append((i + 1, j - 1, 5.0))
                if j + 1 < split:
                    adjacent_indices.append((i + 1, j + 1, 3.0))

            if j > 0:   adjacent_indices.append((i, j - 1, 6.0))

            if j + 1 < split: adjacent_indices.append((i, j + 1, 2.0))

            return adjacent_indices

        adj_list = get_adjacent_indices(drow, dcol)
        randchoice = np.random.randint(len(adj_list))
        (drow2, dcol2, direction) = adj_list[randchoice]
        patch2, cord_list2 = self.patch_from_image(image, drow2, dcol2)

        return patch2, direction, cord_list2

    def show_cropped_patches(self, image, patch1, patch2, direction, cord_list1, cord_list2):
        # cordlist->[(row_s,row_e),(col_s,col_e)]
        gridimage_delete_later = image.clone()
        gridimage_delete_later = gridimage_delete_later.view(self.h, self.w)

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
        plt.show()

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        pil_patch1 = transforms.ToPILImage()(patch1)
        plt.imshow(pil_patch1, cmap='Greys_r')
        plt.title('patch1-red')

        plt.subplot(1, 2, 2)
        pil_patch2 = transforms.ToPILImage()(patch2)
        plt.imshow(pil_patch2, cmap='Greys_r')
        plt.title('patch2-blue')
        plt.show()