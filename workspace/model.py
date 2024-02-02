import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    # input:
    # pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    # output:
    # loss -- a single number for the value of the loss function, [1]

    # TODO: write a loss function for SSD
    #
    # For confidence (class labels), use cross entropy (F.cross_entropy)
    # You can try F.binary_cross_entropy and see which loss is better
    # For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    # Note that you need to consider cells carrying objects and empty cells separately.
    # I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    # and reshape box to [batch_size*num_of_boxes, 4].
    # Then you need to figure out how you can get the indices of all cells carrying objects,
    # and use confidence[indices], box[indices] to select those cells.
    has_obj = ann_confidence[:, :, :-1].sum(axis=2) > 0
    L_conf = F.cross_entropy(pred_confidence[has_obj], ann_confidence[has_obj]) + 3 * F.cross_entropy(pred_confidence[~has_obj], ann_confidence[~has_obj])
    L_box = F.smooth_l1_loss(pred_box[has_obj], ann_box[has_obj])

    return L_conf + L_box


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        self.class_num = class_num  # num_of_classes, in this assignment, 4: cat, dog, person, background

        # TODO: define layers
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.conv2 = ConvBlock(64, 64, 3, 1, 1)
        self.con2_2 = ConvBlock(64, 64, 3, 1, 1)
        self.conv3 = ConvBlock(64, 128, 3, 2, 1)
        self.conv4 = ConvBlock(128, 128, 3, 1, 1)
        self.conv4_2 = ConvBlock(128, 128, 3, 1, 1)
        self.conv5 = ConvBlock(128, 256, 3, 2, 1)
        self.conv6 = ConvBlock(256, 256, 3, 1, 1)
        self.conv6_2 = ConvBlock(256, 256, 3, 1, 1)
        self.conv7 = ConvBlock(256, 512, 3, 2, 1)
        self.conv8 = ConvBlock(512, 512, 3, 1, 1)
        self.conv8_2 = ConvBlock(512, 512, 3, 1, 1)
        self.conv9 = ConvBlock(512, 256, 3, 2, 1)

        self.conv10 = ConvBlock(256, 256, 1, 1)
        self.conv11 = ConvBlock(256, 256, 3, 2, 1)

        self.branch1_conv1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.branch2_conv1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)

        self.conv12 = ConvBlock(256, 256, 1, 1)
        self.conv13 = ConvBlock(256, 256, 3, 1)

        self.branch1_conv2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.branch2_conv2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)

        self.conv14 = ConvBlock(256, 256, 1, 1)
        self.conv15 = ConvBlock(256, 256, 3, 1)

        self.branch1_conv3 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.branch2_conv3 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)

        self.branch1_conv4 = nn.Conv2d(256, 16, kernel_size=1, stride=1)
        self.branch2_conv4 = nn.Conv2d(256, 16, kernel_size=1, stride=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]

        x = x / 255.0  # normalize image. If you already normalized your input image in the dataloader, remove this line.

        # TODO: define forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2_2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv4_2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv6_2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv8_2(x)
        x = self.conv9(x)

        x1 = self.conv10(x)
        x1 = self.conv11(x1)

        # branch 1
        branch1_1 = self.branch1_conv1(x)
        # reshape branch1 from [batch_size, 16, 10, 10] to [batch_size, 16, 10*10]
        branch1_1 = branch1_1.view(branch1_1.size(0), branch1_1.size(1), -1)

        # branch 2
        branch2_1 = self.branch2_conv1(x)
        # reshape branch2 from [batch_size, 16, 5, 5] to [batch_size, 16, 5*5]
        branch2_1 = branch2_1.view(branch2_1.size(0), branch2_1.size(1), -1)

        # main branch
        x2 = self.conv12(x1)
        x2 = self.conv13(x2)

        # branch 1
        branch1_2 = self.branch1_conv2(x1)
        branch1_2 = branch1_2.view(branch1_2.size(0), branch1_2.size(1), -1)

        # branch 2
        branch2_2 = self.branch2_conv2(x1)
        branch2_2 = branch2_2.view(branch2_2.size(0), branch2_2.size(1), -1)

        # main branch
        x3 = self.conv14(x2)
        x3 = self.conv15(x3)

        # branch 1
        branch1_3 = self.branch1_conv3(x2)
        branch1_3 = branch1_3.view(branch1_3.size(0), branch1_3.size(1), -1)

        # branch 2
        branch2_3 = self.branch2_conv3(x2)
        branch2_3 = branch2_3.view(branch2_3.size(0), branch2_3.size(1), -1)

        # branch 1
        branch1_4 = self.branch1_conv4(x3)
        branch1_4 = branch1_4.view(branch1_4.size(0), branch1_4.size(1), -1)

        # branch 2
        branch2_4 = self.branch2_conv4(x3)
        branch2_4 = branch2_4.view(branch2_4.size(0), branch2_4.size(1), -1)

        # concatenate the branches
        branch1 = torch.cat((branch1_1, branch1_2, branch1_3, branch1_4), dim=2)
        branch1 = branch1.permute(0, 2, 1)
        bboxes = branch1.view(branch1.size(0), -1, 4)

        branch2 = torch.cat((branch2_1, branch2_2, branch2_3, branch2_4), dim=2)
        branch2 = branch2.permute(0, 2, 1)
        branch2 = branch2.view(branch2.size(0), -1, 4)
        confidence = self.softmax(branch2)

        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?

        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        # bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        print('bboxes:', bboxes.size())
        print('confidence:', confidence.size())

        return confidence, bboxes
