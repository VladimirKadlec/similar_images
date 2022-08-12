#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Vladimír Kadlec, vladimirkadlec@gmail.com
#
# Build a Faiss nearest neighbour database from input image vectors

import sys
import os
import logging
import click
import math

import faiss
import numpy as np


##########################
def build_index(predictions, dimension=1280):
    """
    Build a Faiss search index

    Args:
        predictions: numpy array with vectors, <no_images> x 4096
            NOTE: normalize the vectors inplace, i.e. changes the argument
    Returns:
        faiss index
    """

    faiss.normalize_L2(predictions)

    # Default exact search, see:
    # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    # and
    # https://github.com/facebookresearch/faiss/wiki/Faster-search
    # Note, it doesn't matter if we use inner product or L2 here
    # index = faiss.IndexFlatIP(dimension)

    # Product quantization index
    # for 40k coco images the index size is 4.7M vs 644M with the Flat index
    # number of subquantizers
    #
    # m = 16
    # # bits allocated per subquantizer
    # n_bits = 8
    # index = faiss.IndexPQ(dimension, m, n_bits)

    # # 44 seconds for 40k coco dataset and 6 cpu cores
    # index.train(predictions)

    
    # Inverted file with PQ refinement
    # 57 sec training, 8M index size
    nlist = 100
    code_size = 16
    n_bits = 8
    nc = int(math.sqrt(predictions.shape[0]))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nc, code_size, n_bits)
    index.nprobe = 4

    index.train(predictions)

    index.add(predictions)
    return index
        
##########################
@click.command(help=
"""
Build a Faiss nearest neighbour database from input image vectors
"""
, no_args_is_help=True,
context_settings=dict(help_option_names=['-h', '--help']))

@click.option ("-v", "--verbose", default=False, is_flag=True,
                help="Increase the logging level to DEBUG")
@click.option ("-i", "--input_vectors", type=click.Path(exists=True),
                required=True,
                help="Typicaly `xxx.saved_predictions.npy` file.")
@click.option ("-o", "--output_filename", type=click.Path(), required=True,
                help="Must not exist. Save the resulting Faiss index there.")
@click.option ("-d", "--dimension", type=int, default=4096,
                show_default=True,
                help="Dimension of the feature vector, just for check.")

def main (verbose, input_vectors, output_filename, dimension):
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG
    logging.basicConfig (level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info ("Starting: '%s'" % __file__)

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

    predictions = np.load(input_vectors)
    if predictions.shape[1] != dimension:
        logging.error(
            "Input vectors have unexpected size, expected %d, got %d" %
            (dimension, predictions.shape[1])
        )
        return 1

    logging.info("Building index...")
    index = build_index(predictions, dimension=dimension)

    logging.info("Writing index to '%s'" % output_filename)
    faiss.write_index(index, output_filename)

    return 0
    

##########################
#Run main
if __name__ == "__main__":
    sys.exit(main())

