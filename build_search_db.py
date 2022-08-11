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

import faiss
import numpy as np


##########################
def build_index(predictions, dimension=4096):
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
    # https://github.com/facebookresearch/faiss/wiki/Faster-search
    # for more practical indexes.
    # Depends on the data size and memory available.
    index = faiss.IndexFlatL2(dimension)

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

def main (verbose, input_vectors, output_filename):
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG
    logging.basicConfig (level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info ("Starting: '%s'" % __file__)

    # Check we can write the output before the expensive computing
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

    dim = 4096
    predictions = np.load(input_vectors)
    if predictions.shape[1] != dim:
        logging.error(
            "Input vectors have unexpected size, expected %d, got %d" %
            (dim, predictions.shape[1])
        )
        return 1

    logging.info("Building index...")
    index = build_index(predictions, dimension = dim)

    logging.info("Writing index to '%s'" % output_filename)
    faiss.write_index(index, output_filename)

    return 0
    

##########################
#Run main
if __name__ == "__main__":
    sys.exit(main())

