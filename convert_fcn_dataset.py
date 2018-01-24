#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16

sdata_dir = r'E:\data\VOCdevkit\VOC2012'
soutput_dir = r'E:\data'

flags = tf.app.flags
flags.DEFINE_string('data_dir', sdata_dir, 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', soutput_dir, 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()
    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸
        return None

    # Your code here, fill the dict

    filename=data.split('/')
    filename=filename[-1].split('.')[0]
    print("picture name:",filename,"Location:",data)
    feature_dict = {
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/encoded': bytes_feature(encoded_data),
        'image/label': int64_list_feature(encoded_label),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here
    # image_address表示一个元组，里面有两个元素，存放两个路径
    writer = tf.python_io.TFRecordWriter(output_filename)
    k=1
    for image_address in file_pars:
        ''' image_address[0]表示第1个路径，如：'F:\AIEngineer\week8\VOCdevkit\VOC2012/JPEGImages/2011_003255.jpg'
            image_address[1]表示第2个路径，如：'F:\AIEngineer\week8\VOCdevkit\VOC2012/SegmentationClass/2011_003255.png'
        '''
        print("开始第",k,"个")
        k+=1
        tf_example = dict_to_tf_example(image_address[0], image_address[1])
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())
    writer.close()
    pass


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []#len(data)=1464 or 1449
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    print("start...")
    logging.info('Prepare dataset file names')
    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')
    train_files = read_images_names(FLAGS.data_dir, True)  # 压缩包
    val_files = read_images_names(FLAGS.data_dir, False)
    print("start creat train data...")
    create_tf_record(train_output_path, train_files)
    print("start creat val data...")
    create_tf_record(val_output_path, val_files)
    print("Finish!")

if __name__ == '__main__':
    tf.app.run()

