# ademxapp

Visual applications by

The University of Adelaide


[//]: # (## PafeNet: Path Fully Effectuated Networks)


### Steps to use

0. Install [MXNet](https://github.com/dmlc/mxnet).

0. Download the [ILSVRC 2012 classification val set](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar), and put the extracted images into the directory:

    data/ilsvrc12/ILSVRC2012_val/

0. Download the models as below, and put them into the directory:

    models/

0. Run the below command to check the classification performance on the ILSVRC 2012 val set:

    ```bash
    sh tools/ilsvrc-cls_eval.sh
    ```

0. Get the network symbols:

    ```python
    from util.symbol.resnet_v2 import rna_model_a, rna_model_a1
    net = rna_model_a()
    net1 = rna_model_a1()
    ```

0. Get the feature symbols (without the global pooling and the top-most linear classifier):

    ```python
    from util.symbol.resnet_v2 import rna_feat_a, rna_feat_a1
    feat = rna_feat_a()
    feat1 = rna_feat_a1()
    ```


### Pre-trained models

Note: Due to a change of MXNet in padding at pooling layers, some of the computed feature maps in Model A will have different sizes from as stated in our paper. However, this has no effect on Model A1, which always uses convolution layers (instead of pooling layers) for down-sampling. So, in most cases, just use Model A1, which was initialized from Model A, and further tuned for several additional epochs.

    model|top-1|top-5
    :---:|:---:|:---:
    [Model A](https://cdn.rawgit.com/itijyou/ademxapp/master/misc/ilsvrc_model_a.pdf) [aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/V7dncO4H0ijzeRj)|19.20%|4.73%
    Model A1 [aar](https://cloudstor.aarnet.edu.au/plus/index.php/s/NOPhJ247fhVDnZH)|19.54%|4.75%


<!---
### Citation

If you use this code or these models in your research, please cite:

    @Misc{PafeNet.2016.Wu,
        author = {Zifeng Wu and Chunhua Shen and Anton van den Hegel},
        title = {Wider or Deeper: Revisiting the ResNet Model for Visual Recognition},
        year = {2016}
        howpublished = {arXiv:?.?}
    }
-->

