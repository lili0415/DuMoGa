# DuMoGa

## Requirements
* `torch==1.6.0`
* `transformers==3.4.0`
* `pytest==5.3.2`
* `scikit-learn==0.22.1`
* `scipy==1.4.1`
* `nltk==3.4.5`

## Explore our dataset
We release a subset dataset (1000 samples containing). Please feel free to explore it.

Steps to explore the dataset:

1. Download images from MS COCO:   [WebsiteMSCOCO](https://cocodataset.org/#download)

    Please download the 2017 Train, 2017 Val, and 2017 Panoptic Train/Val.

2. The subset dataset is in the folder './benchmark', file named 'sub_ris.json'.

3. The ground truth mask of each sample can be obtained by 'get_mask.py' in the same folder. Remember to set the correct path to panoptic 2017.

The format of the annotation:

```
data:{
    'file_name',
    'sentence',
    'pan_seg_file',
    'id'
}
```
'file_name' denotes the name of the image, 'sentence' denotes the query sentence, 'pan_seg_file' denotes the name of the segmentation image, and 'id' helps to find the target area.


## Steps to train DuMoGa
1. Download the pre-processed data, unzip the file into '.benchmark/':

https://drive.google.com/file/d/1AOT4RMUs6nsV-rSItyTR-PGb8N-m9AhZ/view?usp=share_link

2. Run the scripts below:
```bash
python train.py
