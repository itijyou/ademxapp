# ademxapp

Visual applications by the University of Adelaide

In designing our Model A, we did not over-optimize its structure for efficiency unless it was neccessary, which led us to a high-performance model without non-trivial building blocks. Besides, by doing so, we anticipate this model and its trivial variants to perform well when they are finetuned for new tasks, considering their better spatial efficiency and larger model sizes compared to conventional [ResNet](https://arxiv.org/abs/1512.03385) models.

In this work, we try to find a proper depth for ResNets, without grid-searching the whole space, especially when it is too costly to do so, e.g., on the ILSVRC 2012 classification dataset.
For more details, refer to our report: [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080).

This code is a refactored version of the one that we used in the competition, and has not yet been tested extensively, so feel free to open an issue if you find any problem.

To use, first install [MXNet](https://github.com/dmlc/mxnet).


### Updates

* Recent updates
    + Model A1 trained on Cityscapes
    + Model A1 trained on VOC
    + Training code for semantic image segmentation
    + Training code for image classification on ILSVRC 2012 (Still needs to be evaluated.)

<!---
* Planned
-->

* History
    + Results on VOC using COCO for pre-training
    + Fix the bug in testing resulted from changing the EPS in BatchNorm layers
    + Model A1 for ADE20K trained using the *train* set with testing code
    + Segmentation results with multi-scale testing on VOC and Cityscapes
    + Model A and Model A1 for ILSVRC with testing code
    + Segmentation results with single-scale testing on VOC and Cityscapes

### Image classification

#### Pre-trained models

0. Download the ILSVRC 2012 classification val set [6.3GB](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), and put the extracted images into the directory:
    ```
    data/ilsvrc12/ILSVRC2012_val/
    ```

0. Download the models as below, and put them into the directory:
    ```
    models/
    ```

0. Check the classification performance of pre-trained models on the ILSVRC 2012 val set:
    ```bash
    python iclass/ilsvrc.py --data-root data/ilsvrc12 --output output --batch-images 10 --phase val --weights models/ilsvrc-cls_rna-a_cls1000_ep-0001.params --split val --test-scales 320 --gpus 0 --no-choose-interp-method --pool-top-infer-style caffe
    
    python iclass/ilsvrc.py --data-root data/ilsvrc12 --output output --batch-images 10 --phase val --weights models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --split val --test-scales 320 --gpus 0 --no-choose-interp-method
    ```

Results on the ILSVRC 2012 val set tested with a single scale (320, without flipping):

    model|top-1 error (%)|top-5 error (%)|download
    :---:|:---:|:---:|:---:
    [Model A](https://cdn.rawgit.com/itijyou/ademxapp/master/misc/ilsvrc_model_a.pdf)|19.20|4.73|[aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/V7dncO4H0ijzeRj)
    [Model A1](https://cdn.rawgit.com/itijyou/ademxapp/master/misc/ilsvrc_model_a1.pdf)|19.54|4.75|[aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/NOPhJ247fhVDnZH)

Note: Due to a change of MXNet in padding at pooling layers, some of the computed feature maps in Model A will have different sizes from those stated in our report. However, this has no effect on Model A1, which always uses convolution layers (instead of pooling layers) for down-sampling. So, in most cases, just use Model A1, which was initialized from Model A, and tuned for 45k extra iterations.

#### New models

0. Find a machine with 4 devices, each with at least 11G memories.

0. Download the ILSVRC 2012 classification train set [138GB](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar), and put the extracted images into the directory:
    ```
    data/ilsvrc12/ILSVRC2012_train/
    ```
    with the following structure:
    ```
    ILSVRC2012_train
    |-- n01440764
    |-- n01443537
    |-- ...
    `-- n15075141
    ```

0. Train a new Model A from scratch, and check its performance:
    ```bash
    python iclass/ilsvrc.py --gpus 0,1,2,3 --data-root data/ilsvrc12 --output output --model ilsvrc-cls_rna-a_cls1000 --batch-images 256 --crop-size 224 --lr-type linear --base-lr 0.1 --to-epoch 90 --kvstore local --prefetch-threads 8 --prefetcher process --backward-do-mirror
    
    python iclass/ilsvrc.py --data-root data/ilsvrc12 --output output --batch-images 10 --phase val --weights output/ilsvrc-cls_rna-a_cls1000_ep-0090.params --split val --test-scales 320 --gpus 0
    ```

0. Tune a Model A1 from our released Model A, and check its performance:
    ```bash
    python iclass/ilsvrc.py --gpus 0,1,2,3 --data-root data/ilsvrc12 --output output --model ilsvrc-cls_rna-a1_cls1000_from-a --batch-images 256 --crop-size 224 --weights models/ilsvrc-cls_rna-a_cls1000_ep-0001.params --lr-type linear --base-lr 0.01 --to-epoch 9 --kvstore local --prefetch-threads 8 --prefetcher process --backward-do-mirror
    
    python iclass/ilsvrc.py --data-root data/ilsvrc12 --output output --batch-images 10 --phase val --weights output/model ilsvrc-cls_rna-a1_cls1000_from-a_ep-0009.params --split val --test-scales 320 --gpus 0
    ```

0. Or train a new Model A1 from scratch, and check its performance:
    ```bash
    python iclass/ilsvrc.py --gpus 0,1,2,3 --data-root data/ilsvrc12 --output output --model ilsvrc-cls_rna-a1_cls1000 --batch-images 256 --crop-size 224 --lr-type linear --base-lr 0.1 --to-epoch 90 --kvstore local --prefetch-threads 8 --prefetcher process --backward-do-mirror
    
    python iclass/ilsvrc.py --data-root data/ilsvrc12 --output output --batch-images 10 --phase val --weights output/ilsvrc-cls_rna-a1_cls1000_ep-0090.params --split val --test-scales 320 --gpus 0
    ```

It cost more than 40 days on our workstation with 4 Maxwell GTX Titan cards. So, be patient or try smaller models as described in our report.

Note: The best setting (*prefetch-threads* and *prefetcher*) for efficiency can vary depending on the circumstances (the provided CPUs, GPUs, and filesystem).

Note: This code may not accurately reproduce our reported results, since there are subtle differences in implementation, e.g., different cropping strategies, interpolation methods, and padding strategies.


### Semantic image segmentation

We show the effectiveness of our models (as pre-trained features) by semantic image segmenatation using **plain dilated FCNs** initialized from our models. Several A1 models tuned on the *train* set of PASCAL VOC, Cityscapes and ADE20K are available.

* To use, download and put them into the directory:

    ```
    models/
    ```

#### PASCAL VOC 2012:

0. Download the PASCAL VOC 2012 dataset [2GB](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), and put the extracted images into the directory:
    ```
    data/VOCdevkit/VOC2012
    ```
    with the following structure:
    ```
    VOC2012
    |-- JPEGImages
    |-- SegmentationClass
    `-- ...
    ```

0. Check the performance of the pre-trained models:
    ```bash
    python issegm/voc.py --data-root data/VOCdevkit --output output --phase val --weights models/voc_rna-a1_cls21_s8_ep-0001.params --split val --test-scales 500 --test-flipping --gpus 0
    
    python issegm/voc.py --data-root data/VOCdevkit --output output --phase val --weights models/voc_rna-a1_cls21_s8_coco_ep-0001.params --split val --test-scales 500 --test-flipping --gpus 0
    ```

Results on the *val* set:

    model|training data|testing scale|mean IoU (%)|download
    :---|:---:|:---:|:---:|:---:
    Model A1, 2 conv.|VOC; SBD|500|80.84|[aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/YqNptRcboMD44Kd)
    Model A1, 2 conv.|VOC; SBD; COCO|500|82.86|[aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/JKWePbLPlpfRDW4)

Results on the *test* set:

    model|training data|testing scale|mean IoU (%)
    :---|:---:|:---:|:---:
    Model A1, 2 conv.|VOC; SBD|500|[82.5](http://host.robots.ox.ac.uk:8080/anonymous/H0KLZK.html)
    Model A1, 2 conv.|VOC; SBD|multiple|[83.1](http://host.robots.ox.ac.uk:8080/anonymous/BEWE9S.html)
    Model A1, 2 conv.|VOC; SBD; COCO|multiple|[84.9](http://host.robots.ox.ac.uk:8080/anonymous/JU1PXP.html)

#### Cityscapes:

0. Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/downloads/), and put the extracted images into the directory:
    ```
    data/cityscapes
    ```
    with the following structure:
    ```
    cityscapes
    |-- gtFine
    `-- leftImg8bit
    ```

0. Clone the official Cityscapes toolkit:
    ```bash
    git clone https://github.com/mcordts/cityscapesScripts.git data/cityscapesScripts
    ```

0. Check the performance of the pre-trained model:
    ```bash
    python issegm/voc.py --data-root data/cityscapes --output output --phase val --weights models/cityscapes_rna-a1_cls19_s8_ep-0001.params --split val --test-scales 2048 --test-flipping --gpus 0
    ```

0. Tune a Model A1, and check its performance:
    ```bash
    python issegm/voc.py --gpus 0,1,2,3 --split train --data-root data/cityscapes --output output --model cityscapes_rna-a1_cls19_s8 --batch-images 16 --crop-size 500 --origin-size 2048 --scale-rate-range 0.7,1.3 --weights models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --lr-type fixed --base-lr 0.0016 --to-epoch 140 --kvstore local --prefetch-threads 8 --prefetcher process --cache-images 0 --backward-do-mirror

    python issegm/voc.py --gpus 0,1,2,3 --split train --data-root data/cityscapes --output output --model cityscapes_rna-a1_cls19_s8_x1-140 --batch-images 16 --crop-size 500 --origin-size 2048 --scale-rate-range 0.7,1.3 --weights output/cityscapes_rna-a1_cls19_s8_ep-0140.params --lr-type linear --base-lr 0.0008 --to-epoch 64 --kvstore local --prefetch-threads 8 --prefetcher process --cache-images 0 --backward-do-mirror

    python issegm/voc.py --data-root data/cityscapes --output output --phase val --weights output/cityscapes_rna-a1_cls19_s8_x1-140_ep-0064.params --split val --test-scales 2048 --test-flipping --gpus 0
    ```

Results on the *val* set:

    model|training data|testing scale|mean IoU (%)|download
    :---|:---:|:---:|:---:|:---:
    Model A1, 2 conv.|fine|1024x2048|78.08|[aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/2hbvpro6J4XKVIu)

Results on the *test* set:

    model|training data|testing scale|class IoU (%)|class iIoU (%)| category IoU (%)| category iIoU(%)
    :---|:---:|:---:|:---:|:---:|:---:|:---:
    Model A2, 2 conv.|fine|1024x2048|78.4|59.1|90.9|81.1
    Model A2, 2 conv.|fine|multiple|79.4|58.0|91.0|80.1
    Model A2, 2 conv.|fine; coarse|1024x2048|79.9|59.7|91.2|80.8
    Model A2, 2 conv.|fine; coarse|multiple|80.6|57.8|91.0|79.1

For more information, refer to the official [leaderboard](https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results).

Note: [Model A2](https://cdn.rawgit.com/itijyou/ademxapp/master/misc/places_model_a2.pdf) was initialized from Model A, and tuned for 45k extra iterations using the Places data in ILSVRC 2016.

#### MIT Scene Parsing Benchmark (ADE20K):

0. Download the [MIT Scene Parsing dataset](http://sceneparsing.csail.mit.edu/), and put the extracted images into the directory:
    ```
    data/ade20k/
    ```
    with the following structure:
    ```
    ade20k
    |-- annotations
    |   |-- training
    |   `-- validation
    `-- images
        |-- testing
        |-- training
        `-- validation
    ```

0. Check the performance of the pre-trained model:
    ```bash
    python issegm/voc.py --data-root data/ade20k --output output --phase val --weights models/ade20k_rna-a1_cls150_s8_ep-0001.params --split val --test-scales 500 --test-flipping --test-steps 2 --gpus 0
    ```

Results on the *val* set:

    model|testing scale|pixel accuracy (%)|mean IoU (%)|download
    :---|:---:|:---:|:---:|:---:
    [Model A1, 2 conv.](https://cdn.rawgit.com/itijyou/ademxapp/master/misc/ade20k_model_a1.pdf)|500|80.55|43.34|[aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/E4JeZpmssK50kpn)


### Citation

If you use this code or these models in your research, please cite:

    @Misc{word.zifeng.2016,
        author = {Zifeng Wu and Chunhua Shen and Anton van den Hengel},
        title = {Wider or Deeper: {R}evisiting the ResNet Model for Visual Recognition},
        year = {2016}
        howpublished = {arXiv:1611.10080}
    }


### License

This code is only for academic purpose. For commercial purpose, please contact us.


### Acknowledgement

This work is supported with supercomputing resources provided by the PSG cluster at NVIDIA and the Phoenix HPC service at the University of Adelaide.

