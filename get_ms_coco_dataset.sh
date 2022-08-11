#!/bin/bash
# 
# Author: Vladimír Kadlec, vladimirkadlec@gmail.com
#

DIR='ms_coco'
DATASET='val2014.zip'

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
if [ -f ${DATASET} ] ; then
    log_error "File ${DATASET} exists, exitting..."
    exit 1
fi
wget "http://images.cocodataset.org/zips/${DATASET}"
unzip ${DATASET}
