#!/bin/bash
# 
# Download vectors (numpy matrix) for COCO 2014 Validation images
#
# Author: Vladimír Kadlec, vladimirkadlec@gmail.com
#

# Exit when any command fails.
set -e

PREDICTIONS='ms_coco_predictions.zip'

#################
log() {
	>&2 echo "$@"
}

#################
log_error() {
	>&2 echo "ERROR:"
	>&2 echo "$@"
}

if [ -f ${PREDICTIONS} ] ; then
    log_error "File ${PREDICTIONS} exists, not downloading..."
    exit 1
fi

wget 'https://www.dropbox.com/s/zlt6aszbgzgi6d2/ms_coco_predictions_vgg.zip?dl=0'\
    -O ${PREDICTIONS}

log "Note, you don't have to extract 'ms_coco_file_list.txt' if you have it already."
unzip ${PREDICTIONS}

log "All done."
