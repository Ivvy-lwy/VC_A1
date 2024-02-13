import numpy as np
import cv2
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
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                   pass
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                pass
    
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
    cv2.imwrite(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]].jpg",image)



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
    default_output = []
    while len(box_) > 0:
        # 1. Select the bounding box in A with the highest probability in class cat, dog or person.
        max_ = np.max(confidence_, axis=1)

        # 2. If that highest probability is greater than a threshold (threshold=0.5), proceed; otherwise, the NMS is done.
        box_ = box_[max_ > threshold]
        boxs_default = boxs_default[max_ > threshold]

        # 3. Denote the bounding box with the highest probability as x. Move x from A to B.
        x = box_[np.argmax(max_)]
        default_ = boxs_default[np.argmax(max_)]
        output.append(x)
        default_output.append(default_)
        box_ = np.delete(box_, np.argmax(max_), axis=0)

        # 4. For all boxes in A, if a box has IOU greater than an overlap threshold (overlap=0.5) with x, remove that box from A.
        for i in range(len(x)):
            x_min, y_min, x_max, y_max = x[i]
            if iou(box_, x_min, y_min, x_max, y_max) > overlap:
                box_ = np.delete(box_, i, axis=0)
                boxs_default = np.delete(boxs_default, i, axis=0)

    # concatenate the output
    output = np.concatenate(output, axis=0)
    default_output = np.concatenate(default_output, axis=0)

    return output, default_output


def generate_mAP(pred_confidence, pred_box, ann_confidence, ann_box):
    # TODO: Generate mean average precision
    # Plot the precision-recall curve for each class
    # Calculate the area under the curve for each class
    # Return the mean average precision

    # input:
    # pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]

    all_positives = ann_confidence.sum(axis=0)









