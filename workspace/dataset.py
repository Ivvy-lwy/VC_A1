import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2


def default_box_generator(layers, large_scale, small_scale):
    """
    Generate default bounding boxes for all cells in all layers.
    input:
    layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].

    output:
    boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    create an numpy array "boxes" to store default bounding boxes
    you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    the first dimension means number of cells, 10*10+5*5+3*3+1*1
    the second dimension 4 means each cell has 4 default bounding boxes.
    their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    """
    box_num = (10 * 10 + 5 * 5 + 3 * 3 + 1 * 1) * 4
    boxes = np.zeros([box_num, 8])
    idx = 0

    for i, layer in enumerate(layers):
        ssize = small_scale[i]
        lsize = large_scale[i]
        lsize_sqrt2 = lsize * np.sqrt(2)
        lsize_div_sqrt2 = lsize / np.sqrt(2)
        for j in range(layer):
            for k in range(layer):
                # the size of the image is 1
                x_center = (0.5 + k) / layer
                y_center = (0.5 + j) / layer

                # small box
                boxes[idx] = [x_center, y_center, ssize, ssize, x_center - ssize / 2, y_center - ssize / 2,
                              x_center + ssize / 2, y_center + ssize / 2]
                # large box
                boxes[idx + 1] = [x_center, y_center, lsize, lsize, x_center - lsize / 2, y_center - lsize / 2,
                                  x_center + lsize / 2, y_center + lsize / 2]
                # large box
                boxes[idx + 2] = [x_center, y_center, lsize_sqrt2, lsize_div_sqrt2, x_center - lsize_sqrt2 / 2,
                                  y_center - lsize_div_sqrt2 / 2, x_center + lsize_sqrt2 / 2,
                                  y_center + lsize_div_sqrt2 / 2]
                # large box
                boxes[idx + 3] = [x_center, y_center, lsize_div_sqrt2, lsize_sqrt2, x_center - lsize_div_sqrt2 / 2,
                                  y_center - lsize_sqrt2 / 2, x_center + lsize_div_sqrt2 / 2,
                                  y_center + lsize_sqrt2 / 2]
                idx += 4
    # clip the value of the boxes to make sure it is in [0,1]
    boxes = np.clip(boxes, 0.0, 1.0)

    # Visualize the four default bounding boxes for the ith cell
    # print(boxes[0])
    # img = np.ones([300, 300, 3])
    # idx = 0
    # img = cv2.rectangle(img, (int(boxes[idx][4] * 300), int(boxes[idx][5] * 300)),
    #                     (int(boxes[idx][6] * 300), int(boxes[idx][7] * 300)), (0, 0, 255), 2)
    # img = cv2.rectangle(img, (int(boxes[idx + 1][4] * 300), int(boxes[idx + 1][5] * 300)),
    #                     (int(boxes[idx + 1][6] * 300), int(boxes[idx + 1][7] * 300)), (0, 0, 255), 2)
    # img = cv2.rectangle(img, (int(boxes[idx + 2][4] * 300), int(boxes[idx + 2][5] * 300)),
    #                     (int(boxes[idx + 2][6] * 300), int(boxes[idx + 2][7] * 300)), (0, 0, 255), 2)
    # img = cv2.rectangle(img, (int(boxes[idx + 3][4] * 300), int(boxes[idx + 3][5] * 300)),
    #                     (int(boxes[idx + 3][6] * 300), int(boxes[idx + 3][7] * 300)), (0, 0, 255), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return boxes


# this is an example implementation of IOU.
# It is different from the one used in YOLO, please pay attention.
# you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)

    # output:
    # ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]

    inter = np.maximum(np.minimum(boxs_default[:, 6], x_max) - np.maximum(boxs_default[:, 4], x_min), 0) * np.maximum(
        np.minimum(boxs_default[:, 7], y_max) - np.maximum(boxs_default[:, 5], y_min), 0)
    area_a = (boxs_default[:, 6] - boxs_default[:, 4]) * (boxs_default[:, 7] - boxs_default[:, 5])
    area_b = (x_max - x_min) * (y_max - y_min)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def match(ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max):
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box

    # compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)

    ious_true = ious > threshold
    # update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    # if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    # this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence

    if np.sum(ious_true) == 0:
        ious_true = np.argmax(ious)

    # visualize the default bounding boxes that are used to update ann_box and ann_confidence and the ground truth bounding box
    # img = np.ones([300, 300, 3])
    # for i in range(len(boxs_default[ious_true])):
    #     img = cv2.rectangle(img, (int(boxs_default[ious_true][i][4] * 300), int(boxs_default[ious_true][i][5] * 300)),
    #                         (int(boxs_default[ious_true][i][6] * 300), int(boxs_default[ious_true][i][7] * 300)), (0, 0, 255), 2)
    # img = cv2.rectangle(img, (int(x_min * 300), int(y_min * 300)),
    #                     (int(x_max * 300), int(y_max * 300)), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


    # make sure at least one default bounding box is used
    # update ann_box and ann_confidence (do the same thing as above)
    ann_confidence[ious_true, cat_id] = 1
    ann_confidence[ious_true, -1] = 0
    # Compute center, width and height of the ground truth bounding box
    G_x = (x_min + x_max) / 2
    G_y = (y_min + y_max) / 2
    G_w = x_max - x_min
    G_h = y_max - y_min
    # update ann_box
    ann_box[ious_true, 0] = (G_x - boxs_default[ious_true, 0]) / boxs_default[ious_true, 2]  # (G_x - P_x) / P_w
    ann_box[ious_true, 1] = (G_y - boxs_default[ious_true, 1]) / boxs_default[ious_true, 3]  # (G_y - P_y) / P_h
    ann_box[ious_true, 2] = np.log(G_w / boxs_default[ious_true, 2])  # log(G_w / P_w)
    ann_box[ious_true, 3] = np.log(G_h / boxs_default[ious_true, 3])  # log(G_h / P_h)


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train=True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num

        # overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)

        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        # notice:
        # you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        self.img_names = self.img_names[:int(len(self.img_names) * 0.9)] if self.train else self.img_names[int(len(
            self.img_names) * 0.9):]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)  # one-hot vectors
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background

        ann_confidence[:, -1] = 1  # the default class for all cells is set to "background"

        img_name = self.imgdir + self.img_names[index]
        ann_name = self.anndir + self.img_names[index][:-3] + "txt"

        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        # 3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        # 4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.

        # to use function "match":
        # match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        # where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.

        # note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        # For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)

        image = cv2.imread(img_name)
        # image_cpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(ann_name, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                class_id = int(line[0])
                x_min = float(line[1]) / image.shape[1]
                y_min = float(line[2]) / image.shape[0]
                x_max = (float(line[1]) + float(line[3])) / image.shape[1]
                y_max = (float(line[2]) + float(line[4])) / image.shape[0]
                match(ann_box, ann_confidence, self.boxs_default, self.threshold, class_id, x_min, y_min, x_max, y_max)

                # Visualize the ground truth bounding boxes
                # cv2.rectangle(image_cpy, (int(x_min * image.shape[1]), int(y_min * image.shape[0])),
                #               (int(x_max * image.shape[1]), int(y_max * image.shape[0])), (0, 255, 0), 2)

        # Visualize the image, and ground truth bounding boxes

        # plt.imshow(image_cpy)
        # plt.show()

        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.transpose((2, 0, 1))

        # TODO: Data augmentation
        # if self.train:
        #     # Random cropping
        #     image, ann_box, ann_confidence = random_crop(image, ann_box, ann_confidence)

        # image = torch.from_numpy(image).float()
        # ann_box = torch.from_numpy(ann_box).float()
        # ann_confidence = torch.from_numpy(ann_confidence).float()

        return image, ann_box, ann_confidence


def random_crop(image, ann_box, ann_confidence, crop_size=256):
    """
    Randomly crop the image and bounding boxes
    image: [3, 320, 320]
    ann_box: [num_of_boxes, 4]
    ann_confidence: [num_of_boxes, num_of_classes]
    crop_size: int
    return: image, ann_box, ann_confidence
    """

    image = image.transpose((1, 2, 0))
    height, width, _ = image.shape
    # print(height, width)
    # print(ann_box)
    # print(ann_confidence)

    # Randomly select a crop center
    center_x = np.random.randint(crop_size // 2, width - crop_size // 2)
    center_y = np.random.randint(crop_size // 2, height - crop_size // 2)

    # Crop the image
    image = image[center_y - crop_size // 2:center_y + crop_size // 2,
            center_x - crop_size // 2:center_x + crop_size // 2, :]
    # print(image.shape)

    # Crop the bounding boxes
    ann_box[:, 0] = ann_box[:, 0] * width - center_x + crop_size // 2
    ann_box[:, 1] = ann_box[:, 1] * height - center_y + crop_size // 2
    ann_box[:, 2] = ann_box[:, 2] * width - center_x + crop_size // 2
    ann_box[:, 3] = ann_box[:, 3] * height - center_y + crop_size // 2
    # print(ann_box)

    # Remove the bounding boxes that are out of the image
    ann_box = ann_box[
        (ann_box[:, 0] >= 0) & (ann_box[:, 1] >= 0) & (ann_box[:, 2] <= crop_size) & (ann_box[:, 3] <= crop_size)]
    # print(ann_box)

    # Remove the bounding boxes that are too small
    ann_box = ann_box[(ann_box[:, 2] - ann_box[:, 0]) > 5]
    ann_box = ann_box[(ann_box[:, 3] - ann_box[:, 1]) > 5]
    # print(ann_box)

    # Remove the bounding boxes that are too large
    ann_box = ann_box[(ann_box[:, 2] - ann_box[:, 0]) < crop_size - 5]
    ann_box = ann_box[(ann_box[:, 3] - ann_box[:, 1]) < crop_size - 5]
    # print(ann_box)

    return image.transpose((2, 0, 1)), ann_box, ann_confidence


if __name__ == "__main__":
    boxs_default = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])
    box_num = 3
    class_num = 4
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=True, image_size=320)
    dataset.__getitem__(9)
