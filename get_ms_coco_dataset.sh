#!/bin/bash
# 
# Download the COCO 2014 Validation images
#
# Author: Vladimír Kadlec, vladimirkadlec@gmail.com
#

# Exit when any command fails.
set -e

DIR='ms_coco'
IMG_DIR="${DIR}/val2014"
DATASET='val2014.zip'
IMG_LIST='ms_coco_file_list.txt'

#################
log() {
	>&2 echo "$@"
}

#################
log_error() {
	>&2 echo "ERROR:"
	>&2 echo "$@"
}

mkdir -p ${DIR}

cd ${DIR}
if [ ! -f ${DATASET} ] ; then
    wget "http://images.cocodataset.org/zips/${DATASET}"
    unzip ${DATASET}
else
    log "File ${DATASET} exists, not downloading..."
fi
cd ..
if [ ! -f ${IMG_LIST} ] ; then
    ls ${IMG_DIR} > ${IMG_LIST}
else
    log "Filelist ${IMG_LIST} exists, not listing..."
fi
log "All done."
