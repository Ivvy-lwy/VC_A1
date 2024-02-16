import numpy as np
import cv2
from matplotlib import pyplot as plt

from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                #image1: draw ground truth bounding boxes on image1
                relative_x, relative_y, relative_w, relative_h = ann_box[i]
                default_x, default_y, default_w, default_h = boxs_default[i][0:4]
                gt_x = relative_x * default_w + default_x
                gt_y = relative_y * default_h + default_y
                gt_w = default_w * np.exp(relative_w)
                gt_h = default_h * np.exp(relative_h)
                x1 = int((gt_x - gt_w/2)*image.shape[1])
                y1 = int((gt_y - gt_h/2)*image.shape[0])
                x2 = int((gt_x + gt_w/2)*image.shape[1])
                y2 = int((gt_y + gt_h/2)*image.shape[0])
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                image1 = cv2.rectangle(image1, (x1, y1), (x2, y2), color, thickness)

                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                default_x1, default_y1, default_x2, default_y2 = boxs_default[i][4:8]
                default_x1 = int(default_x1*image.shape[1])
                default_y1 = int(default_y1*image.shape[0])
                default_x2 = int(default_x2*image.shape[1])
                default_y2 = int(default_y2*image.shape[0])
                image2 = cv2.rectangle(image2, (default_x1, default_y1), (default_x2, default_y2), color, thickness)

    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                relative_x, relative_y, relative_w, relative_h = pred_box[i]
                default_x, default_y, default_w, default_h = boxs_default[i][0:4]
                pd_x = relative_x * default_w + default_x
                pd_y = relative_y * default_h + default_y
                pd_w = default_w * np.exp(relative_w)
                pd_h = default_h * np.exp(relative_h)
                x1 = int((pd_x - pd_w/2)*image.shape[1])
                y1 = int((pd_y - pd_h/2)*image.shape[0])
                x2 = int((pd_x + pd_w/2)*image.shape[1])
                y2 = int((pd_y + pd_h/2)*image.shape[0])
                color = colors[j]
                thickness = 2
                image3 = cv2.rectangle(image3, (x1, y1), (x2, y2), color, thickness)

                # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                default_x1, default_y1, default_x2, default_y2 = boxs_default[i][4:8]
                default_x1 = int(default_x1*image.shape[1])
                default_y1 = int(default_y1*image.shape[0])
                default_x2 = int(default_x2*image.shape[1])
                default_y2 = int(default_y2*image.shape[0])
                image4 = cv2.rectangle(image4, (default_x1, default_y1), (default_x2, default_y2), color, thickness)

    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)

    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    cv2.imwrite(windowname+".jpg",image)



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.

    output = []
    output_confidence = []
    default_output = []
    ids = []
    while len(box_) > 0:
        # 1. Select the bounding box in A with the highest probability in class cat, dog or person.
        max_ = np.max(confidence_, axis=1)

        # 2. If that highest probability is greater than a threshold (threshold=0.5), proceed; otherwise, the NMS is done.
        box_high = box_[max_ > threshold]

        if len(box_high) == 0:
            break

        # 3. Denote the bounding box with the highest probability as x. Move x from A to B.
        max_id = np.argmax(max_)
        ids.append(max_id)
        x_high = box_[max_id]
        default_ = boxs_default[max_id]
        output.append(x_high)
        default_output.append(default_)
        box_ = np.delete(box_, max_id, axis=0)
        boxs_default = np.delete(boxs_default, max_id, axis=0)

        conf = confidence_[np.argmax(max_)]
        output_confidence.append(conf)
        confidence_ = np.delete(confidence_, max_id, axis=0)

        # 4. For all boxes in A, if a box has IOU greater than an overlap threshold (overlap=0.5) with x, remove that box from A.
        # breakpoint()
        x, y, w, h = default_[0:4]
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        ious = iou(boxs_default, x_min, y_min, x_max, y_max)
        # delete the boxes with iou > overlap
        box_ = np.delete(box_, np.where(ious > overlap), axis=0)
        confidence_ = np.delete(confidence_, np.where(ious > overlap), axis=0)
        boxs_default = np.delete(boxs_default, np.where(ious > overlap), axis=0)

    # concatenate the output
    if len(output) == 0:
        print("No bounding box found")
        # breakpoint()
        return np.array([]), np.array([]), np.array([]), np.array([])
    output = np.concatenate(output, axis=0)
    default_output = np.concatenate(default_output, axis=0)
    output_confidence = np.concatenate(output_confidence, axis=0)
    ids = np.array(ids)

    return output, default_output, output_confidence, ids


def update_precision_recall(pred_confidence_, pred_box_, ann_confidence_, ann_box_, boxs_default,thres):
    """Update the precision and recall for each class"""
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(pred_confidence_)):
        pred_confidence = pred_confidence_[i]
        pred_box = pred_box_[i]
        ann_confidence = ann_confidence_[i]
        ann_box = ann_box_[i]

        pred_box, default_box, pred_confidence, ids = non_maximum_suppression(pred_confidence, pred_box, boxs_default, threshold=thres)
        for j in range(len(ids)):
            # TODO: compare default with ann_box (don't know index in ann_box)
            x, y, w, h = ann_box[j].flatten()
            x_min = x - w / 2
            y_min = y - h / 2
            x_max = x + w / 2
            y_max = y + h / 2
            default_box = default_box.reshape(-1, 8)
            ious = iou(default_box, x_min, y_min, x_max, y_max)
            true_positive += np.sum(ious > thres)
            false_positive += len(ious) - true_positive
        false_negative += len(np.where(ann_confidence>0)) - true_positive

    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    return precision, recall


def generate_mAP(precision, recall, epoch):
    # TODO: Generate mean average precision
    # Plot the precision-recall curve for each class
    # Calculate the area under the curve for each class
    # Return the mean average precision

    # Plot the precision-recall curve for each class
    # Store the plot as a .png file
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve_epoch_{}.png'.format(epoch))










