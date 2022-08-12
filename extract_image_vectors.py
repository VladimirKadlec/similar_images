#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Vladimír Kadlec, vladimirkadlec@gmail.com
#
# Extract feature vector from the input list of images.
# VGG16 model pre-trained on ImageNet is used.

import sys
import os
import logging
import click

import tensorflow as tf
# The following block limits number of CPU cores used by tensorflow.
# Use for machines with limited resources. Note, that the number can't be
# less than 2.
######
num_threads = 4
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)
#######

#from tensorflow.keras.applications.efficientnet import EfficientNetB0
#from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing import image

from keras import models, Model
import tensorflow.keras.models
import numpy as np


##########################
def init_model(model_filename='vgg_4096.h5'):
    """
    Return pre-trained model used for image predictions.
    Loads the model from the file, If it doesn't exist create it
    from the tensorflow installation a save it there.

    Args:
        model_filename: Filename for the model.
    Returns:
        keras Model
    """
    model = None
    try:
        model = tensorflow.keras.models.load_model(
            model_filename, compile=False)
    except OSError:
        #effnet = EfficientNetB0(weights='imagenet', include_top=True)
        #model = Model(effnet.input, effnet.layers[-3].output)
        vgg = VGG16(weights='imagenet', include_top=True)
        model = Model(vgg.input, vgg.layers[-2].output)
        model.save(model_filename)
    return model

##########################
def image_list_to_vectors(model, img_dir='ms_coco/val2014',
    file_list = '', batch_size = 32):
    """
    Convert images from 'file_list' to feature vector by prediction
    from the EfficientNetB0 'model'.

    Args:
        model: Keras Model for predictions.
        img_dir: Directory containing the input images.
        file_list: List of image filenames (withoud the dir name).
        batch_size: Predict batch_size images in one batch, depends
                    on RAM (or GPU RAM) available.
    Returns:
        (filename_list, predictions)
        where:
            filename_list: list of filenames
            predictions: numpy.ndarray, len(filename_list) x dim
                where dim is 4096 for VGG16 and 1280 for efficientnet
    """

    assert(batch_size > 0)
    batch_list = []
    batch_filename_list = []
    res = None
    cur_index = 0
    fileno = 0
    with open(file_list) as f:
        for filename in f:
            fileno += 1
            if cur_index > batch_size-1:
                logging.info(
                    "Predicting... %d files processed so far" % (fileno-1))
                features = model.predict_on_batch(np.stack(batch_list))
                if res is None:
                    res = features
                else:
                    res = np.concatenate([res, features])
                batch_list = []
                act_result = None
                cur_index = 0
            filename = filename.strip('\n')
            img = image.load_img(os.path.join(img_dir, filename), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            batch_list.append(x[0])
            batch_filename_list.append(filename)
            cur_index += 1
    if batch_list:
        features = model.predict_on_batch(np.stack(batch_list))
        if res is None:
            res = features
        else:
            res = np.concatenate([res, features])
    return batch_filename_list, res
        
##########################
@click.command(help=
"""
Extract feature vector from the input list of images. The images are read from
the `img_dir` directory. VGG16 model pre-trained on ImageNet is used.
"""
, no_args_is_help=True,
context_settings=dict(help_option_names=['-h', '--help']))

@click.option ("-v", "--verbose", default=False, is_flag=True,
                help="Increase the logging level to DEBUG")
@click.option ("-l", "--filename_list", type=click.Path(), required=True)
@click.option ("-o", "--output_filename", type=click.Path(), default=None,
                help="Must not exist. Save the resulting vectors there. The default output filename is: '`filename_list`.saved_predictions.npy'")
@click.option ("-m", "--model_filename", type=click.Path(),
                default='vgg_4096.h5', show_default=True,
                help="If the file doesn't exist the model is downloaded and saved there. Load the model from the file otherwise."
                )
@click.option ("-d", "--img_dir", type=click.Path(exists=True),
                default='ms_coco/val2014', show_default=True)
@click.option ("-b", "--batch_size", type=int, default=32, show_default=True)


def main (verbose, model_filename, filename_list, img_dir, batch_size,
    output_filename):
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG
    logging.basicConfig (level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info ("Starting: '%s'" % __file__)

    if output_filename is None:
        output_filename = filename_list + '.saved_predictions.npy'
    # Check we can write the output before doing the expensive computing
    try:
        with open(output_filename, 'r'):
            logging.error(
                "Output filename '%s' exists." % output_filename)
            return 1
    except:
        pass
    try:
        with open(output_filename, 'a') as f:
            pass
        os.unlink(output_filename)
    except:
        logging.error(
            "Output filename '%s' is not writeable." % output_filename)
        return 1

    model = init_model(model_filename)
    fl, predictions = image_list_to_vectors(model, file_list=filename_list,
        img_dir=img_dir, batch_size=batch_size)

    logging.info("Saving the vectors to '%s'" % output_filename)
    np.save(output_filename, predictions)

    logging.info("All done.")
    return 0
    

##########################
#Run main
if __name__ == "__main__":
    sys.exit(main())

