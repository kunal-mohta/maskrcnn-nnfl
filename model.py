import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import skimage.transform
import scipy

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

#Macros
IMG_MIN_DIM = 800
IMG_MAX_DIM = 1024
IMG_SHAPE = np.array([IMG_MAX_DIM, IMG_MAX_DIM, 3])
MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
NUM_CLASSES = 1 + 80 # We should reduce this
IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + NUM_CLASSES
ANCHOR_STRIDE_RPN = 1
ANCHOR_RATIOS_RPN = [0.5, 1, 2]
ANCHOR_SCALES_RPN = (32, 64, 128, 256, 512)
PYRAMID_SIZE = 256
POST_NMS_ROIS_TRAINING = 2000
POST_NMS_ROIS_INFERENCE = 1000
NMS_THRESHOLD_RPN = 0.7
PRE_NMS_LIMIT = 6000
IMAGES_PER_GPU = 1
DETECTION_MAX_INSTANCES = 100
DETECTION_MIN_CONFIDENCE = 0.7
DETECTION_NMS_THRESHOLD = 0.3
RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
TRAIN_ROIS_PER_IMAGE = 200
ROI_POSITIVE_RATIO = 0.33
MASK_SHAPE = [28, 28]
IMAGE_SHAPE = np.array([1024, 1024, 3])
BACKBONE_STRIDES = [4, 8, 16, 32, 64]
BACKBONE = "resnet101"
BATCH_SIZE = 1
POOL_SIZE = 7
MASK_POOL_SIZE = 14
USE_MINI_MASK = True
MINI_MASK_SHAPE = (56, 56)
NAME="coco"
LEARNING_RATE = 0.001
LEARNING_MOMENTUM = 0.9
VALIDATION_STEPS = 50
RPN_TRAIN_ANCHORS_PER_IMAGE = 256
GRADIENT_CLIP_NORM = 5.0
LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
WEIGHT_DECAY = 0.0001


#  Utility Functions

def custom_print(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


def compute_backbone_shapes(image_shape):
    return np.array([[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))] for stride in BACKBONE_STRIDES])


def norm_boxes(boxes, shape):
    """Converts boxes to normalized coordinates from pixel coordinates.
    shape: in pixels
    boxes: in pixel coordinates
    """
    h, w = shape
    shift = np.array([0, 0, 1, 1])
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    nor_coordinates = np.divide((boxes - shift), scale).astype(np.float32)
    return nor_coordinates


def denorm_boxes(boxes, shape):
    """Converts boxes to pixel coordinates from normalized coordinates.
    Basically oposite of norm_boxes fucntion
    boxes: in normalized coordinates
    shape: in pixels
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    pixel_coordinates = np.around(np.multiply(boxes, scale) + shift).astype(np.int32)
    return  pixel_coordinates 

def resize_image(image):
    """Images are resized to 'square' of size [IMG_MAX_DIM, IMG_MAX_DIM]
    without changing the aspect ratio.
    """

    # Keeping track of the image dtype, to return results in the same dtype
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    
    min_image_dim = IMG_MIN_DIM
    max_image_dim = IMG_MAX_DIM

    # Scaling to min_image_dim(if >max(h,w)) before to max_image_dim
    if min_image_dim:
        # Only upscaling
        scale = max(1, min_image_dim / min(h, w))

    # Should not exceed maximum dimension
    if max_image_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_image_dim:
            scale = max_image_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
                            image, (round(h * scale), round(w * scale)),
                            order=1, mode='constant', cval=0, clip=True,
                            preserve_range=True, anti_aliasing=False,
                            anti_aliasing_sigma=None)

    # Need padding
    # Get new height and width
    h, w = image.shape[:2]
    top_padding = (max_image_dim - h) // 2
    bottom_padding = max_image_dim - h - top_padding
    left_padding = (max_image_dim - w) // 2
    right_padding = max_image_dim - w - left_padding
    padding = [(top_padding, bottom_padding), (left_padding, right_padding), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    # New window 
    window = (top_padding, left_padding, h + top_padding, w + left_padding)
    
    return image.astype(image_dtype), window, scale, padding


#  Bounding box Utility Functions

def extract_bboxes(mask):
    """ Computes bounding boxes from masks """
    
    bboxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for x in range(mask.shape[-1]):
        m = mask[:, :, x]
        # Bounding box.
        indicies_horizontal = np.where(np.any(m, axis=0))[0]
        indicies_vertical = np.where(np.any(m, axis=1))[0]
        if indicies_horizontal.shape[0]:
            x1, x2 = indicies_horizontal[[0, -1]]
            y1, y2 = indicies_vertical[[0, -1]]
            # x2 and y2 incremented by 1 bcz they shouldn't be part of bbox.
            x2 = x2 + 1
            y2 = y2 + 1
        else:
            # Setting bbox to zero as no mask for this instance. 
            # This can happen due to cropping or resizing.
            x2, x1, y2, y1 = 0, 0, 0, 0
        bboxes[x] = np.array([y1, x1, y2, x2])
    return bboxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU (Intersection over union) of given box 
    with the list of given boxes.
    """
    # Computing areas of intersection
    y1 = np.maximum(boxes[:, 0], box[0])
    x1 = np.maximum(boxes[:, 1], box[1])
    y2 = np.minimum(boxes[:, 2], box[2])
    x2 = np.minimum(boxes[:, 3], box[3])
    intersection_area = np.maximum(y2 - y1, 0) * np.maximum(x2 - x1, 0)
    # Computing areas of union
    union_area = box_area + boxes_area[:] - intersection_area[:]
    iou = intersection_area / union_area   # IoU is simply an evalutaion metric. 
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU (Intersection over Union) overlaps between two sets of boxes.
    """
    x1_1 = boxes1[:, 1]
    x2_1 = boxes1[:, 3]
    y1_1 = boxes1[:, 0]
    y2_1 = boxes1[:, 2]
    x1_2 = boxes2[:, 1]
    x2_2 = boxes2[:, 3]
    y1_2 = boxes2[:, 0]
    y2_2 = boxes2[:, 2]
    
    # Areas of  GT boxes and anchores.
    area1 = (y2_1 - y1_1) * (x2_1 - x1_1)
    area2 = (y2_2 - y1_2) * (x2_2 - x1_2)

    # Calculates overlaps to create matrix [boxes1_count, boxes2_count]
    # Each cell of the matrix contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for x in range(overlaps.shape[1]):
        box = boxes2[x]
        box_area = area2[x]
        overlaps[:, x] = compute_iou(box, boxes1, box_area, area1)
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Calculates IoU (Intersection over union) overlaps between two sets of masks. """
    
    # Return empty resuly if any of the masks is empty
    if masks2.shape[-1] == 0 or masks1.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks (reshape) and calculate their areas
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area2 = np.sum(masks2, axis=0)
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    
    # IoU
    intersections_area = np.dot(masks1.T, masks2)
    union_area = area1[:, None] + area2[None, :] - intersections_area
    iou = intersections_area / union_area

    return iou


def box_refinement_graph(box, gt_box):
    """Calculates all the refinements needed to transform box to gt_box """
    
    # casting to float
    gt_box = tf.cast(gt_box, tf.float32)
    box = tf.cast(box, tf.float32)

    # computing dimensions 
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_x_center = gt_box[:, 1] + 0.5 * gt_width
    gt_y_center = gt_box[:, 0] + 0.5 * gt_height
    
    width = box[:, 3] - box[:, 1]
    height = box[:, 2] - box[:, 0]   
    x_center = box[:, 1] + 0.5 * width
    y_center = box[:, 0] + 0.5 * height
    
    dw = tf.log(gt_width / width)
    dh = tf.log(gt_height / height)
    dx = (gt_x_center - x_center) / width
    dy = (gt_y_center - y_center) / height
    
    res = tf.stack([dy, dx, dh, dw], axis=1)
    return res


def box_refinement(box, gt_box):
    """Calculates all the refinements required to transform box to gt_box. """

    gt_box = gt_box.astype(np.float32)
    box = box.astype(np.float32)

    # computing dimensions 
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_x_center = gt_box[:, 1] + 0.5 * gt_width
    gt_y_center = gt_box[:, 0] + 0.5 * gt_height
    
    width = box[:, 3] - box[:, 1]
    height = box[:, 2] - box[:, 0]   
    x_center = box[:, 1] + 0.5 * width
    y_center = box[:, 0] + 0.5 * height

    dw = np.log(gt_width / width)
    dh = np.log(gt_height / height)
    dx = (gt_x_center - x_center) / width
    dy = (gt_y_center - y_center) / height
    
    res = np.stack([dy, dx, dh, dw], axis=1)
    return res


#  Data Utility Functions

def resize_mask(mask, padding, scale):
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def minimize_mask(mask, bbox, minimized_shape):   
    """
    Mask is shrunk to reduce load on the memory. 
    """
    minimized_mask = np.zeros(minimized_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        y1, x1, y2, x2 = bbox[i][:4]
        m = mask[:, :, i].astype(bool)
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("ounding box area zero")
        # Using bilinear interpolation to resize
        m = skimage.transform.resize(
                        m, minimized_shape,order=1, mode='constant', 
                        cval=0, clip=True, preserve_range=False, 
                        anti_aliasing=False, anti_aliasing_sigma=None)
        minimized_mask[:, :, i] = np.around(m).astype(np.bool)
    return minimized_mask
    
def unmold_mask(mask, bbox, image_shape):
    """Converts the mask outputted by the neural network to its original shape.
    A binary mask of the original image size.
    """
    y1, x1, y2, x2 = bbox
    threshold = 0.5
    mask = skimage.transform.resize(
                        mask, (y2 - y1, x2 - x1),order=1,mode='constant',
                        cval=0, clip=True,preserve_range=False,
                        anti_aliasing=False, anti_aliasing_sigma=None)
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Aligning the mask to correct location
    final_mask = np.zeros(image_shape[:2], dtype=np.bool)
    final_mask[y1:y2, x1:x2] = mask
    return final_mask


#  Resnet Graph

def conv_block(input_tensor, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=True)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x)
    #x = BatchNorm(name=bn_name_base + '2a')(x, training=False)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filter2, (3, 3), padding='same', name=conv_name_base + '2b', use_bias=True)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x)
    #x = BatchNorm(name=bn_name_base + '2b')(x, training=False)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filter3, (1, 1), name=conv_name_base + '2c', use_bias=True)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x)
    #x = BatchNorm(name=bn_name_base + '2c')(x, training=False)

    # conv block at shortcut
    shortcut = KL.Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=True)(input_tensor)
    shortcut = KL.BatchNormalization(name=bn_name_base + '1')(shortcut)
    #shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=False)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name=str(stage) + block + '_out')(x)
    return x

def identity_block(input_tensor, filters, stage, block):
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(filter1, (1, 1), name=conv_name_base + '2a',use_bias=True)(input_tensor)
    x = KL.BatchNormalization(name=bn_name_base + '2a')(x) #changed
    #x = BatchNorm(name=bn_name_base + '2a')(x, training=False)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filter2, (3, 3), padding='same',name=conv_name_base + '2b', use_bias=True)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2b')(x)
    #x = BatchNorm(name=bn_name_base + '2b')(x, training=False)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(filter3, (1, 1), name=conv_name_base + '2c',use_bias=True)(x)
    x = KL.BatchNormalization(name=bn_name_base + '2c')(x)
    #x = BatchNorm(name=bn_name_base + '2c')(x, training=False)

    shortcut = input_tensor
    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name=str(stage) + block + '_out')(x)
    return x

def resnet(input_image):
    # 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization(name='bn_conv1')(x)
    #x = BatchNorm(name='bn_conv1')(x, training=False)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # 2
    x = conv_block(x, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, [64, 64, 256], stage=2, block='c')
    # 3
    x = conv_block(x, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, [128, 128, 512], stage=3, block='d')
    # 4
    x = conv_block(x, [256, 256, 1024], stage=4, block='a')
    block_count = 22 # for resnet101
    for i in range(block_count):
        x = identity_block(x, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # 5
    x = conv_block(x, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, [512, 512, 2048], stage=5, block='b')
    C5 = x = identity_block(x, [512, 512, 2048], stage=5, block='c')

    return [C1, C2, C3, C4, C5]

def fpn(resnet_feature_maps):

    _, C2, C3, C4, C5 = resnet_feature_maps

    P5 = KL.Conv2D(256, (1,1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name='fpn_p4add')([KL.Conv2D(256, (1,1), name='fpn_c4p4')(C4), KL.UpSampling2D(size=(2,2), name='fpn_c5upsampled')(P5)])
    P3 = KL.Add(name='fpn_p3add')([KL.Conv2D(256, (1,1), name='fpn_c3p3')(C3), KL.UpSampling2D(size=(2,2), name='fpn_c4upsampled')(P4)])
    P2 = KL.Add(name='fpn_p2add')([KL.Conv2D(256, (1,1), name='fpn_c2p2')(C2), KL.UpSampling2D(size=(2,2), name='fpn_c3upsampled')(P3)])
    
    P5 = KL.Conv2D(256, (3,3), padding='same', name='fpn_p5')(P5)
    P6 = KL.MaxPooling2D(pool_size=(1,1), strides=(2,2), name='fpn_p6')(P5)
    P4 = KL.Conv2D(256, (3,3), padding='same', name='fpn_p4')(P4)
    P3 = KL.Conv2D(256, (3,3), padding='same', name='fpn_p3')(P3)
    P2 = KL.Conv2D(256, (3,3), padding='same', name='fpn_p2')(P2)

    return [P2, P3, P4, P5, P6]

def generate_anchors(image_shape):

    # This function generates anchor boxes for all layers of pyramid

    # An anchor is represented by (y1,x1,y2,x2)
    # (y1,x1) is upper left corner
    # (y2,x2) is bottom right corner
    
    backbone_shapes = compute_backbone_shapes(image_shape)
    anchors = []   # [anchor_count, (y1, x1, y2, x2)]

    # Cache anchors and reuse if image shape is the same
    #if not hasattr(self, "_anchor_cache"):
    #anchor_cache = {}
    #if not tuple(image_shape) in anchor_cache:

    for i in range(len(ANCHOR_SCALES_RPN)):
        scale = ANCHOR_SCALES_RPN[i]
        feature_stride = BACKBONE_STRIDES[i]
        feature_shape = backbone_shapes[i]
        ratio = ANCHOR_RATIOS_RPN
        anchor_stride = ANCHOR_STRIDE_RPN
        
        # Get all combinations of scales and ratios
        scale, ratio = np.meshgrid(np.array(scale), np.array(ratio))
        scale = scale.flatten()
        ratio = ratio.flatten()

        # Enumerate heights and widths from scales and ratios
        height = scale / np.sqrt(ratio)
        width = scale * np.sqrt(ratio)

        # Enumerate shifts in feature space
        shift_y = np.arange(0, feature_shape[0], anchor_stride) * feature_stride
        shift_x = np.arange(0, feature_shape[1], anchor_stride) * feature_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        # Enumerate combinations of shifts, widths, and heights
        box_width, box_center_x = np.meshgrid(width, shift_x)
        box_height, box_center_y = np.meshgrid(height, shift_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_center = np.stack([box_center_y, box_center_x], axis=2).reshape([-1, 2])
        box_size = np.stack([box_height, box_width], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_center - 0.5 * box_size, box_center + 0.5 * box_size], axis=1)
        anchors.append(boxes)

    anchors = np.concatenate(anchors, axis=0)
    return norm_boxes(anchors, image_shape[:2])
    #anchor_cache[tuple(image_shape)] = norm_boxes(anchors, image_shape[:2])
    #return anchor_cache[tuple(image_shape)]
    

#  Proposal Layer

def apply_box_deltas(boxes, deltas):
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes(boxes, window):
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    
    def __init__(self, proposal_count, nms_threshold, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        print("scores=", scores)
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]
        pre_nms_limit = tf.minimum(PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        print("ix=", ix)
        ix = tf.squeeze(ix, [0])
        scores = tf.squeeze(scores, [0])
        scores = tf.gather(scores,ix)
        #print("scores=", scores)
        #scores = KL.Lambda(lambda t: tf.expand_dims(t, 0))(scores)
        scores = tf.expand_dims(scores, 0)
        print("scores=", scores)
        '''
        print("deltas=", deltas)
        print("deltas[0,1]=", deltas[0,1])
        print("deltas[1]=", deltas[1])
        print("deltas[2]=", deltas[2])
        print("deltas[1:]=", deltas[1:])
        print("deltas[0:2]=", deltas[0:2])
        #print("deltas_slice[1]=", tf.slice(deltas, )
        '''
        deltas = tf.squeeze(deltas, [0])
        #print("deltas=", deltas)
        #deltas = KL.Lambda(lambda t: tf.gather(t[0], t[1]))([deltas, ix])
        deltas = tf.gather(deltas, ix)
        #print("deltas=", deltas)
        #deltas = KL.Lambda(lambda t: tf.expand_dims(t, 0))(deltas)
        deltas = tf.expand_dims(deltas, 0)
        print("deltas=", deltas) 
        #print("anchors=", anchors)
        anchors = tf.squeeze(anchors, [0])
        #anchors = KL.Lambda(lambda t: tf.gather(t[0], t[1]))([anchors, ix])
        anchors = tf.gather(anchors, ix)
        #print("anchors=", anchors)
        #pre_nms_anchors = KL.Lambda(lambda t: tf.expand_dims(t, 0))(anchors)  
        pre_nms_anchors = tf.expand_dims(anchors, 0)
        print("pre_nms_anchors=",pre_nms_anchors)

        pre_nms_anchors = tf.squeeze(pre_nms_anchors, [0])
        deltas = tf.squeeze(deltas, [0])
        #boxes = KL.Lambda(lambda t: apply_box_deltas_graph(t[0], t[1]))([pre_nms_anchors, deltas])
        #boxes = KL.Lambda(lambda t: tf.expand_dims(t, 0))(boxes)
        boxes = apply_box_deltas(pre_nms_anchors, deltas)
        boxes = tf.expand_dims(boxes, 0)
        print("boxes=", boxes)
        window = np.array([0,0,1,1], dtype=np.float32)
        #window = tf.convert_to_tensor(window, dtype=tf.float32)
        boxes = tf.squeeze(boxes, [0])
        #boxes = KL.Lambda(lambda t: clip_boxes_graph(t[0], t[1]))([boxes, window])
        #boxes = KL.Lambda(lambda t: tf.expand_dims(t, 0))(boxes)
        boxes = clip_boxes(boxes, window)
        boxes = tf.expand_dims(boxes, 0)
        print("boxes=", boxes)
        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            #print("indices=",indices)
            #print("proposal=", proposals)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        boxes = tf.squeeze(boxes, [0])
        scores = tf.squeeze(scores, [0])
        #proposals = KL.Lambda(lambda t: nms(t[0], t[1]))([boxes, scores])
        #proposals = KL.Lambda(lambda t: tf.expand_dims(t, 0))(proposals)
        proposals = nms(boxes, scores)
        proposals = tf.expand_dims(proposals, 0)
        print("proposals=",proposals)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


#  Detection Target Layer

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks):
    # Assertions
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(TRAIN_ROIS_PER_IMAGE * ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    #if USE_MINI_MASK:
    # Transform ROI coordinates from normalized image space
    # to normalized mini-mask space.
    y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
    gt_h = gt_y2 - gt_y1
    gt_w = gt_x2 - gt_x1
    y1 = (y1 - gt_y1) / gt_h
    x1 = (x1 - gt_x1) / gt_w
    y2 = (y2 - gt_y1) / gt_h
    x2 = (x2 - gt_x1) / gt_w
    boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes, box_ids, MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    
    def __init__(self, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        proposals = tf.squeeze(proposals, [0])
        gt_class_ids = tf.squeeze(gt_class_ids, [0])
        gt_boxes = tf.squeeze(gt_boxes, [0])
        gt_masks = tf.squeeze(gt_masks, [0])
        outputs = detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks)
        #print(type(outputs))
        outputs = list(outputs)
        #print(type(outputs))
        #print(type(outputs[0]))
        #print(outputs[1])
        #tf.cast(outputs[1], tf.float32)
        #print(outputs[1])
        outputs[0] = tf.expand_dims(outputs[0], 0)
        outputs[1] = tf.expand_dims(outputs[1], 0)
        outputs[2] = tf.expand_dims(outputs[2], 0)
        outputs[3] = tf.expand_dims(outputs[3], 0)
        #tf.cast(outputs[1], tf.int32)
        #print(outputs[1])
        #outputs = tuple(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, TRAIN_ROIS_PER_IMAGE, MASK_SHAPE[0], MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


#  Detection Layer

def refine_detections_graph(rois, probs, deltas, window):
   
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas(
        rois, deltas_specific * BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes(refined_rois, window)

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=DETECTION_MAX_INSTANCES,
                iou_threshold=DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):

    def __init__(self, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        rois = tf.squeeze(rois, [0])
        mrcnn_class = tf.squeeze(mrcnn_class, [0])
        mrcnn_bbox = tf.squeeze(mrcnn_bbox, [0])
        window = tf.squeeze(window, [0])
        detections_batch = refine_detections_graph(rois, mrcnn_class, mrcnn_bbox, window)
        detections_batch = tf.expand_dims(detections_batch, 0)

        return tf.reshape(detections_batch, [BATCH_SIZE, DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, DETECTION_MAX_INSTANCES, 6)


#  Region Proposal Network (RPN)

def region_proposal_network(feature_map):

    shared = KL.Conv2D(512, (3,3), padding='same', activation='relu', name='rpn_conv_shared')(feature_map)
    #shared = KL.Activation('relu')(shared)   #shape = batch,w,h,512

    # There are 3 anchors per pixel, each with different aspect ratios. 
    # Different scales are used for different levels of pyramid.
    # Hence, only one scale of anchors is used in 1 level of pyramid. 
    # Each anchor has two probabilities associated with it: One for foreground(object) and other for background
    # Therefore, there will be six channels
    x = KL.Conv2D(6, (1,1), name='rpn_class_raw')(shared)         # shape = batch,w,h,6
    #shape_x = tf.shape(x)
    class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x) #shape = batch,w*h*3,2
    class_probs = KL.Activation('softmax', name="rpn_class_xxx")(class_logits)

    # A bounding box can be defined by 4 coordinates. For each pixel, there are 3 anchors and for each anchor,
    # 4 values representing a bounding box. Convolution of shared layer should result in 12 channels
    x = KL.Conv2D(12, (1,1), name='rpn_bbox_pred')(shared)       # shape = batch, w, h, 12
    #shape_x = tf.shape(x)
    bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)   #shape = batch, w*h*3, 4

    return [class_logits, class_probs, bbox]

def build_rpn():
    rpn_input = KL.Input(shape=[None, None, 256], name="input_rpn_feature_map")
    rpn_output = region_proposal_network(rpn_input)
    return KM.Model([rpn_input], rpn_output, name="rpn_model")
    
    
#  ROIAlign Layer

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


#  Feature Pyramid Network Heads

def fpn_classifier(rois, feature_maps, image_meta, num_classes):

    #x = KL.Lambda(lambda t: roi_align(t))([rois, image_meta, tf.convert_to_tensor(POOL_SIZE)] + feature_maps)
    x = PyramidROIAlign([POOL_SIZE, POOL_SIZE], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(1024, (POOL_SIZE, POOL_SIZE), padding="valid"), name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn1')(x)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn2')(x)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

def fpn_mask(rois, feature_maps, image_meta, num_classes):

    # ROI Pooling
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    #x = KL.Lambda(lambda t: roi_align(t))([rois, image_meta, tf.convert_to_tensor(MASK_POOL_SIZE)] + feature_maps)
    x = PyramidROIAlign([MASK_POOL_SIZE, MASK_POOL_SIZE], name="roi_align_mask")([rois, image_meta] + feature_maps)
    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn2')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn3')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn4')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
    return x


#  Loss Functions

def smooth_l1_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(target_bbox, rpn_match, rpn_bbox):
    
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0, smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0, K.binary_crossentropy(target=y_true, output=y_pred), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


#  Data Generator

def load_image_gt(dataset, image_id, augment=False, augmentation=None, use_mini_mask=False):
    
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding = resize_image(image)
    mask = resize_mask(mask, padding, scale)
    
    # Random horizontal flips.
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(mask, bbox, MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask

def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois

def create_anchors(anchor_scales, anchor_ratios, shape, feature_stride, anchor_stride):
    anchor_scales, anchor_ratios = np.meshgrid(np.array(anchor_scales), np.array(anchor_ratios))
    anchor_scales = anchor_scales.flatten()
    anchor_ratios = anchor_ratios.flatten()

    heights = anchor_scales / np.sqrt(anchor_ratios)
    widths = anchor_scales * np.sqrt(anchor_ratios)

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

def data_generator(ds, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    img_ind = -1
    img_ids = np.copy(ds.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # anchors
    bkbn_strides = [4, 8, 16, 32, 64]
    anchor_scales = (32, 64, 128, 256, 512)
    anchor_ratios = [0.5, 1, 2]
    anchor_stride = 1
    bkbn_shapes = np.array(
                    [[int(math.ceil(IMAGE_SHAPE[0] / stride)),
                        int(math.ceil(IMAGE_SHAPE[1] / stride))]
                        for stride in bkbn_strides])
    anchors = []
    for i in range(len(anchor_scales)):
        anchors.append(create_anchors(anchor_scales[i], anchor_ratios, bkbn_shapes[i], bkbn_strides[i], anchor_stride))
    anchors = np.concatenate(anchors, axis=0)

    rpn_anchors_count = 256
    max_ground_truths = 100
    batch_index = 0
    while True:
        try:
            # next image pick
            img_ind = (img_ind + 1) % len(img_ids)
            if shuffle and img_ind == 0:
                np.random.shuffle(img_ids)

            # ground truth bboxs and masks
            image_id = img_ids[img_ind]
            if ds.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(ds, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(ds, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=USE_MINI_MASK)

            if not np.any(gt_class_ids > 0):
                continue

            image_shape = image.shape

            # exclude coco crowds
            crowd_cls_ids = np.where(gt_class_ids < 0)[0]
            if crowd_cls_ids.shape[0] > 0:
                non_crowd_cls_ids = np.where(gt_class_ids > 0)[0]
                boxes = gt_boxes[crowd_cls_ids] # crowd boxes
                gt_class_ids = gt_class_ids[non_crowd_cls_ids]
                gt_boxes = gt_boxes[non_crowd_cls_ids]
                overlaps = compute_overlaps(anchors, boxes)
                max_iou = np.amax(overlaps, axis=1)
                no_crowd_bool = (max_iou < 0.001)
            else:
                no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

            overlaps = compute_overlaps(anchors, gt_boxes)

            rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
            rpn_bbox = np.zeros((RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

            anchor_iou_argmax = np.argmax(overlaps, axis=1)
            anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
            
            # negatives
            rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1 # iou of anchor with ground truth < 0.3

            # positives
            gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis = 0))[: ,0]
            rpn_match[gt_iou_argmax] = 1
            rpn_match[anchor_iou_max >= 0.7] = 1 # iou of anchor with ground truth >= 0.7 is positive

            pos_ids = np.where(rpn_match == 1)[0] # positive ids
            extra_anchors = len(pos_ids) - (rpn_anchors_count // 2) # positives not more than half
            if extra_anchors > 0:
                # make extra anchors neutral
                neutral_ids = np.random.choice(pos_ids, extra_anchors, replace=False)
                rpn_match[neutral_ids] = 0

            # similiarly, for negatives
            neg_ids = np.where(rpn_match == -1)[0]
            extra_anchors = len(neg_ids) - (rpn_anchors_count - np.sum(rpn_match == 1))
            if extra_anchors > 0:
                # make extra anchors neutral
                neutral_ids = np.random.choice(neg_ids, extra_anchors, replace=False)
                rpn_match[neutral_ids] = 0

            # For positive anchors, compute shift and scale needed to transform them
            # to match the corresponding GT boxes.
            pos_ids = np.where(rpn_match == 1)[0]
            ind = 0  # rpn_bbox index
            for anchor, pos_id in zip(anchors[pos_ids], pos_ids):
                # Closest gt box (it might have IoU < 0.7)
                ground = gt_boxes[anchor_iou_argmax[pos_id]]

                ground_h = ground[2] - ground[0]
                ground_w = ground[3] - ground[1]
                ground_center_y = ground[0] + 0.5 * ground_h
                ground_center_x = ground[1] + 0.5 * ground_w

                anch_h = anchor[2] - anchor[0]
                anch_w = anchor[3] - anchor[1]
                anch_center_y = anchor[0] + 0.5 * anch_h
                anch_center_x = anchor[1] + 0.5 * anch_w

                # ground truth bbox refinement (should come from rpn)
                rpn_bbox[ind] = [
                    (ground_center_y - anch_center_y) / anch_h,
                    (ground_center_x - anch_center_x) / anch_w,
                    np.log(ground_h / anch_h),
                    np.log(ground_w / anch_w),
                ]
                rpn_bbox[ind] /= np.array([0.1, 0.1, 0.2, 0.2]) # normalize
                ind += 1

            # return rpn_match, rpn_bbox


            # initialization
            if batch_index == 0:
                batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros([batch_size, rpn_anchors_count, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, max_ground_truths), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, max_ground_truths, 4), dtype=np.int32)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1], max_ground_truths), dtype=gt_masks.dtype)
                
            # remove extra
            if gt_boxes.shape[0] > max_ground_truths:
                selected_ids = np.random.choice(np.arange(gt_boxes.shape[0]), max_ground_truths, replace=False)
                gt_class_ids = gt_class_ids[selected_ids]
                gt_boxes = gt_boxes[selected_ids]
                gt_masks = gt_masks[:, :, selected_ids]

            batch_image_meta[batch_index] = image_meta
            batch_rpn_match[batch_index] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[batch_index] = rpn_bbox
            batch_images[batch_index] = mold_image(image.astype(np.float32))
            batch_gt_class_ids[batch_index, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[batch_index, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[batch_index, :, :, :gt_masks.shape[-1]] = gt_masks

            
            batch_index += 1

            if batch_index >= batch_size: # full batch
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                yield inputs, outputs # bcux generator

                # new batch
                batch_index = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(ds.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


#  MaskRCNN Class
class MaskRCNN():
    def __init__(self, mode, model_dir):
        self.mode = mode
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode)

    def build(self, mode):
        # print(" ")
        # print("123456789")
        # print("qwertyuiop")
        # print(" ")
        print("changes here 123")
        # Inputs
        input_image = KL.Input(shape=[None, None, IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[IMAGE_META_SIZE], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            #if USE_MINI_MASK:
            input_gt_masks = KL.Input(shape=[MINI_MASK_SHAPE[0], MINI_MASK_SHAPE[1], None], name="input_gt_masks", dtype=bool)
            
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
        
        resnet_outputs = resnet(input_image)
        _, C2, C3, C4, C5 = resnet_outputs
        
        fpn_outputs = fpn(resnet_outputs)
        rpn_feature_maps = fpn_outputs
        mrcnn_feature_maps = fpn_outputs[:-1]
        head_inputs = mrcnn_feature_maps
        
        if mode == "training":
            anchors = generate_anchors(IMAGE_SHAPE)
            anchors = np.broadcast_to(anchors, (BATCH_SIZE,) + anchors.shape)
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors
        
        rpn_model = build_rpn()
        rpn_outputs = []
        for feature_map in fpn_outputs:
            #output = rpn_model([feature_map])
            rpn_outputs.append(rpn_model([feature_map]))

        # Convert from list of lists of level outputs to list of lists of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*rpn_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]
        rpn_class_logits, rpn_class, rpn_bbox = outputs
        
        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        if mode == "training":
            proposal_count = POST_NMS_ROIS_TRAINING
        else:
            proposal_count = POST_NMS_ROIS_INFERENCE
        
        rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=NMS_THRESHOLD_RPN, name="ROI")([rpn_class, rpn_bbox, anchors])
        
        #rpn_rois = propose(proposal_count, NMS_THRESHOLD_RPN, rpn_class, rpn_bbox, anchors)
        #rpn_rois = KL.Lambda(propose)([proposal_count, rpn_class, rpn_bbox, anchors, NMS_THRESHOLD_RPN])
        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
            target_rois = rpn_rois
            
            # Generate detection targets
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(name="proposal_targets")([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier(rois, mrcnn_feature_maps, input_image_meta, NUM_CLASSES)
            mrcnn_mask = fpn_mask(rois, mrcnn_feature_maps, input_image_meta, NUM_CLASSES)
            
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(*x), name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")([target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")([target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")([target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier(rpn_rois, mrcnn_feature_maps, input_image_meta, NUM_CLASSES)

            # Detections
            detections = DetectionLayer(name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            
            mrcnn_mask = fpn_mask(detection_boxes, mrcnn_feature_maps, input_image_meta, NUM_CLASSES)
            
            inputs = [input_image, input_image_meta, input_anchors]
            outputs = [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox]
            model = KM.Model(inputs, outputs, name='mask_rcnn')

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(errno.ENOENT, "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        
        import h5py
        # Conditional import to support versions of Keras before 2.2
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, clipnorm=GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss", "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True) * LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(layer.output, keepdims=True) * LOSS_WEIGHTS.get(name, 1.))
            # self.keras_model.metrics_tensors.append(loss)
            self.keras_model.metrics.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            custom_print("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        if hasattr(keras_model, "inner_model"):
            layers = keras_model.inner_model.layers
        else:
            layers = keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                custom_print("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None,
              steps_per_epoch=1000):
       
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, shuffle=True, batch_size=BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        custom_print("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        custom_print("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)


    def mold_inputs(self, images):
        
        img_metas = []
        windows = []
        molded_imgs = []
        
        for img in images:
            # Resize image
            molded_img, window, scale, padding = resize_image(img)
            molded_img = mold_image(molded_img)
            # Build image_meta
            img_meta = compose_image_meta(
                0, img.shape, molded_img.shape, window, scale,
                np.zeros([NUM_CLASSES], dtype=np.int32))
            # Append
            molded_imgs.append(molded_img)
            windows.append(window)
            img_metas.append(img_meta)
        # Pack into arrays
        molded_imgs = np.stack(molded_imgs)
        img_metas = np.stack(img_metas)
        windows = np.stack(windows)
        return molded_imgs, img_metas, windows


    def unmold_detections(self, detections, maskrcnn_mask, original_img_shape, img_shape, window):
        
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        boxes = detections[:N, :4]
        masks = maskrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = norm_boxes(window, img_shape[:2])
        w_y1, w_x1, w_y2, w_x2 = window
        shift = np.array([w_y1, w_x1, w_y1, w_x1])
        w_w = w_x2 - w_x1  # width of window
        w_h = w_y2 - w_y1  # height of window
        scale = np.array([w_h, w_w, w_h, w_w]) 
        # Converting the boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Converting the boxes to pixel coordinates on the original image
        boxes = denorm_boxes(boxes, original_img_shape[:2])

        # Excluding (filtering out) detections with zero area. 
        exclude = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude.shape[0] > 0:
            boxes = np.delete(boxes, exclude, axis=0)
            class_ids = np.delete(class_ids, exclude, axis=0)
            scores = np.delete(scores, exclude, axis=0)
            masks = np.delete(masks, exclude, axis=0)
            N = class_ids.shape[0]

        # Resizing masks to original image size
        full_size_masks = []
        for i in range(N):
            # Converting neural network masks to mask of full size
            full_size_mask = unmold_mask(masks[i], boxes[i], original_img_shape)
            full_size_masks.append(full_size_mask)
        full_size_masks = np.stack(full_size_masks, axis=-1)\
            if full_size_masks else np.empty(original_img_shape[:2] + (0,))

        return boxes, class_ids, scores, full_size_masks


    def detect(self, images, verbose=0):
        assert len(images) == BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            custom_print("Processing {} images".format(len(images)))
            for x in images:
                custom_print("image", x)

        # Inputs required to be molded to the format which is expected by neural network
        molded_imgs, img_metas, windows = self.mold_inputs(images)

        img_shape = molded_imgs[0].shape    # Since all image sizes are same
      
        # Anchors
        anchors = generate_anchors(img_shape)
        # Duplicating (because keras requires it) across batch dimension
        anchors = np.broadcast_to(anchors, (BATCH_SIZE,) + anchors.shape)

        if verbose:
            custom_print("molded_images", molded_imgs)
            custom_print("image_metas", img_metas)
            custom_print("anchors", anchors)
        
        # Running object detection
        detections, _, _, maskrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_imgs, img_metas, anchors], verbose=0)
        # Processing detections
        results = []
        for i, image in enumerate(images):
            rois_final, class_ids_final, scores_final, masks_final =\
                self.unmold_detections(detections[i], maskrcnn_mask[i],
                                       image.shape, molded_imgs[i].shape,
                                       windows[i])
            results.append({
                "rois": rois_final,
                "class_ids": class_ids_final,
                "scores": scores_final,
                "masks": masks_final,
            })
        return results


#  Data Formatting

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(RGB_images):
    
    return RGB_images.astype(np.float32) - MEAN_PIXEL


def unmold_image(normalized_molded_images):
    """Takes a image and returns the original image which was normalized with mold_image""" 
    return (normalized_molded_images + MEAN_PIXEL).astype(np.uint8)


#  Misc
def trim_zeros_graph(boxes, name='trim_zeros'):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
    
