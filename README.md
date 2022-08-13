# Similar images demo application
## Problem definition
A client asks us to create a solution for finding the most similar content. A user uploads an item (e.g. photo), the solution presents the most similar items available to the user

## Solution
In the following we work with **images/photos** as an example of user item. The demonstration application is built in the following steps:
- create an image database
- build a demo application taking an image as an input and returning similar images from the database as an output.

The images from the COCO dataset "2014 Val images" are used to populate the image database, see [COCO website](https://cocodataset.org) for details. Our database contains 40504 images, most of them are photos of real world common objects.

The demo application is a Jupyter Notebook. You can search for random images from the database, or upload your own image as an input. The results are images from the database (i.e. these 40504 images from the COCO dataset).

<img alt="Demo application" src="demo_example.gif" width="700">

## Technical description
### Image database
- The **40504 images** from the COCO dataset, 6.8G of jpeg files.

- We compute a feature vector from each image. Our **feature vector** contains
**4096** 32bit numbers and it is computed by **VGG16** deep neural network
pretrained on ImageNet, see:
[https://keras.io/api/applications/vgg/](https://keras.io/api/applications/vgg/)
The implementation is in [extract_image_vectors.py](extract_image_vectors.py).

- The feature vectors are added to similarity search index, we use Facebook's Faiss library: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss).

There are several options for effective search. The implementation in
[build_search_db.py](build_search_db.py) is based on product quantization, 16
sub-quantizers with 8 bits, 100 centroids. The **resulting index** in just
**8M bytes**.  Note, that the quantization requires training, we use the whole
dataset as the training set, because our database is static.  For the demo
purposes even brute force exact search would be fast enough, as 40k images is
really a small number.

### Demo application
The demo application is a Jupyter Notebook, see [Similar image.ipynb](Similar&#32;image.ipynb).

### Installation
The development was done on `Ubuntu 22.04.1 LTS`, 4 CPU cores, 4G RAM server, in python3.
#### Requirements
You usually use virtualenv + pip, conda or similar to install the following packages:
```
click
tensorflow
pillow
notebook
ipywidgets
faiss-cpu
```
**Note:** You have to enable `widgetsnbextension` before starting jupyter notebook. If you have virtual environment, use:
```
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```
command **before** `jupyter notebook` launch.
#### Steps
The following steps should work on `Ubuntu 22.04.1 LTS` Linux server.
The whole installation requires cca 14G disk space.
Don't forget to **activate your virtual environment** with installed packages.
1. Clone this repo, enter the directory:
```
$git clone 'git@github.com:VladimirKadlec/similar_images.git'
$cd similar_images
```

2. Download and extract MS COCO images (6.2G file, 6.8G extracted size):
```
$./get_ms_coco_dataset.sh
```

3. Download **or** compute feature vectors for images:
- download vectors from my DropBox (225M file, 633M extracted size):
```
$./get_ms_coco_vectors.sh
```
- or compute them on your own, it takes 83 min on 4 core CPU, cca 2G RAM.
```
$./extract_image_vectors.py -l ms_coco_file_list.txt
```

4. Build Faiss index:
The index is built under 50 seconds.
```
$./build_search_db.py -i ms_coco_file_list.txt.saved_predictions.npy -o ms_coco_file_list.faiss.index
```

5. Run Jupyter Notebook with file [Similar image.ipynb](Similar&#32;image.ipynb). You may need to click `Kernel/Restart & Run All` for the first time.

## Discussion
1. How to evaluate the quality of the similar search?
- A/B testing against the current solution. If there is no current solution
  get an explicit user feedback from small number of users (e.g. questionnaire).

2. Is it fast enough for millions of images?
- The feature extraction takes cca 150ms for a single image  + 2ms similarity
  search on CPU. In general, the main slow part is the feature extraction, the
  Faiss library is fast enough.
- It depends on actual setup and requirements. The speed can be improved by:
  - batch processing (2x-3x speedup on CPU with enough memory)
  - use of GPU machines (20x-100x speedup)
  - use of different network for feature extracting, e.g. EfficientNet.
  - use of cluster of machines (linear speedup with respect to number of machines).

3. Feature vector size 4096, isn't it too big?
- The vectors are quantized during indexing by the Faiss library to cca 16 x 8 bits.

4. VGG16 vs EfficientNetB0:
- The EfficientNetB0 is much faster and much smaller, the results from the
  experiments weren't satisfactory.

5. Is it possible to add new images to the index?
- Yes, new images (feature vectors) can be added to the index without
  rebuilding the whole index.

6. What about the similarity of texts?
- See proof of concept of proof of concept in [Text_similarity.ipynb](Text_similarity.ipynb). The database is 20k tweets from Sentiment140 dataset, the index is pre-computed and stored in this repo.
