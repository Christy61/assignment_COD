"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2 as cv
import os
import sys

import numpy as np
import imageio
# import scipy as scp
# import scipy.misc

import argparse

import logging

from convcrf import convcrf
from fullcrf import fullcrf

import torch
from torch.autograd import Variable

from utils import pascal_visualizer as vis
from utils import synthetic

import time

try:
    import matplotlib.pyplot as plt
    matplotlib = True
    figure = plt.figure()
    plt.close(figure)
except:
    matplotlib = False
    pass

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def do_crf_inference(image, unary, args):

    if args.pyinn or not hasattr(torch.nn.functional, 'unfold'):
        # pytorch 0.3 or older requires pyinn.
        args.pyinn = True
        # Cheap and easy trick to make sure that pyinn is loadable.
        import pyinn

    # get basic hyperparameters
    num_classes = unary.shape[2]
    shape = image.shape[0:2]
    config = convcrf.default_conf
    config['filter_size'] = 7
    config['pyinn'] = args.pyinn

    if args.normalize:
        # Warning, applying image normalization affects CRF computation.
        # The parameter 'col_feats::schan' needs to be adapted.

        # Normalize image range
        #     This changes the image features and influences CRF output
        image = image / 255
        # mean substraction
        #    CRF is invariant to mean subtraction, output is NOT affected
        image = image - 0.5
        # std normalization
        #       Affect CRF computation
        image = image / 0.3

        # schan = 0.1 is a good starting value for normalized images.
        # The relation is f_i = image / schan
        config['col_feats']['schan'] = 0.1

    # make input pytorch compatible
    img = image.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to image: [1, 3, height, width]
    img = img.reshape([1, 3, shape[0], shape[1]])
    img_var = Variable(torch.Tensor(img))

    un = unary.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to unary: [1, 21, height, width]
    un = un.reshape([1, num_classes, shape[0], shape[1]])
    unary_var = Variable(torch.Tensor(un))

    logging.debug("Build ConvCRF.")
    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes,
                                use_gpu=not args.cpu)

    # move to GPU if requested
    if not args.cpu:
        img_var = img_var.cuda()
        unary_var = unary_var.cuda()
        gausscrf.cuda()


    # Perform ConvCRF inference
    """
    'Warm up': Our implementation compiles cuda kernels during runtime.
    The first inference call thus comes with some overhead.
    """
    logging.info("Start Computation.")
    prediction = gausscrf.forward(unary=unary_var, img=img_var)

    if args.nospeed:

        logging.info("Doing speed benchmark with filter size: {}"
                     .format(config['filter_size']))
        logging.info("Running multiple iteration. This may take a while.")

        # Our implementation compiles cuda kernels during runtime.
        # The first inference run is those much slower.
        # prediction = gausscrf.forward(unary=unary_var, img=img_var)

        start_time = time.time()
        for i in range(10):
            # Running ConvCRF 10 times and report average total time
            prediction = gausscrf.forward(unary=unary_var, img=img_var)

        prediction.cpu()  # wait for all GPU computations to finish
        duration = (time.time() - start_time) * 1000 / 10

        logging.debug("Finished running 10 predictions.")
        logging.debug("Avg Computation time: {} ms".format(duration))

    # Perform FullCRF inference
    myfullcrf = fullcrf.FullCRF(config, shape, num_classes)
    fullprediction = myfullcrf.compute(unary, image, softmax=False)

    if args.nospeed:

        start_time = time.time()
        for i in range(5):
            # Running FullCRF 5 times and report average total time
            fullprediction = myfullcrf.compute(unary, image, softmax=False)

        fullduration = (time.time() - start_time) * 1000 / 5

        logging.debug("Finished running 5 predictions.")
        logging.debug("Avg Computation time: {} ms".format(fullduration))

        logging.info("Using FullCRF took {:4.0f} ms ({:2.2f} s)".format(
            fullduration, fullduration / 1000))

        logging.info("Using ConvCRF took {:4.0f} ms ({:2.2f} s)".format(
            duration, duration / 1000))

        logging.info("Congratulation. Using ConvCRF provids a speed-up"
                     " of {:.0f}.".format(fullduration / duration))

        logging.info("")

    return prediction.data.cpu().numpy(), fullprediction

def plot_results(image, pros, conv_out, full_out, label, name):
    """
    Plots and saves the results without any mapping. All images are displayed in grayscale.
    """    
    if matplotlib:
        # Plot results using matplotlib
        figure = plt.figure(figsize=(15,3))
        figure.tight_layout()
        # Plot parameters
        num_rows = 1
        num_cols = 5
        off = 0
        output_folder_1 = "./adaboost/conv_crf_output-1"
        output_folder_2 = "./adaboost/full_crf_output-1"
        os.makedirs(output_folder_1, exist_ok=True)
        os.makedirs(output_folder_2, exist_ok=True)
        name = name.replace("jpg", "png")
        
        # Plot original image
        ax = figure.add_subplot(num_rows, num_cols, 1)
        ax.set_title('Image')
        ax.axis('off')
        ax.imshow(image, cmap='gray')

        # Plot label
        ax = figure.add_subplot(num_rows, num_cols, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(label, cmap='gray')

        # Plot unary (convert to grayscale if necessary)
        ax = figure.add_subplot(num_rows, num_cols, 3)
        ax.set_title('Pros')
        ax.axis('off')
        ax.imshow(pros, cmap='gray')

        # Plot ConvCRF output (convert to grayscale if necessary)
        conv_out = conv_out[0]  # Remove Batch dimension
        conv_hard = np.argmax(conv_out, axis=0) if conv_out.ndim == 3 else conv_out
        plt.imsave(os.path.join(output_folder_1, name), conv_hard, cmap='gray')
        
        ax = figure.add_subplot(num_rows, num_cols, 4)
        ax.set_title('ConvCRF Output')
        ax.axis('off')
        ax.imshow(conv_hard, cmap='gray')

        # Plot FullCRF output (convert to grayscale if necessary)
        full_hard = np.argmax(full_out, axis=2) if full_out.ndim == 3 else full_out
        ax = figure.add_subplot(num_rows, num_cols, 5)
        plt.imsave(os.path.join(output_folder_2, name), full_hard, cmap='gray')
        ax.set_title('FullCRF Output')
        ax.axis('off')
        ax.imshow(full_hard, cmap='gray')

        # Save the figure
        # output_path = f"test_output_COMprompter/{name}"
        # plt.savefig(output_path, dpi=300)
    else:
        if args.output is None:
            args.output = "out.png"

        logging.warning("Matplotlib not found.")
        logging.info("Saving output to {} instead".format(args.output))

    return


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("pros_path", type=str,
                        help="pros file.")
    
    parser.add_argument("--gpu", type=str, default='0',
                        help="which gpu to use")

    parser.add_argument('--output', type=str,
                        help="Optionally save output as img.")

    parser.add_argument('--nospeed', action='store_false',
                        help="Skip speed evaluation.")

    parser.add_argument('--normalize', action='store_true',
                        help="Normalize input image before inference.")

    parser.add_argument('--pyinn', action='store_true',
                        help="Use pyinn based Cuda implementation"
                             "for message passing.")

    parser.add_argument('--cpu', action='store_true',
                        help="Run on CPU instead of GPU.")

    return parser

def expand_label_to_two_channels(label):
    if not (0 <= label).all() or not (label <= 1).all():
        raise ValueError("label should in [0, 1] range")
    background = 1 - label
    expanded_label = np.stack((background, label), axis=-1)

    return expanded_label

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    img_dir = "../../dataset/test/image"
    for name in os.listdir(img_dir):
        image_path = img_dir + "/" + name
        label_path = image_path.replace("image", "GT")
        label_path = label_path.replace("jpg", "png")
        pros_path = args.pros_path + "/" + name
        pros_path = pros_path.replace("jpg", "png")
        # Load data
        image = imageio.imread(image_path)
        label = imageio.imread(label_path)
        pros_ = imageio.imread(pros_path)
        H, W = label.shape
        pros_resized = cv.resize(pros_, (W, H), interpolation=cv.INTER_CUBIC)
        pros_resized = pros_resized / 255.0
        pros = expand_label_to_two_channels(pros_resized)
        # Produce unary by adding noise to label
        # unary = synthetic.augment_label(label, num_classes=2)
        # Compute CRF inference
        conv_out, full_out = do_crf_inference(image, pros, args)
        plot_results(image, pros_, conv_out, full_out, label, name)
