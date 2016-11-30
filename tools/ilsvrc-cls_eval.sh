#!/bin/sh

python iclass/ilsvrc.py --data-root data/ilsvrc12 --output output --batch-images 10 --phase val --weight models/ilsvrc-cls_rna-a_cls1000_ep-0001.params --split val --test-scales 320

python iclass/ilsvrc.py --data-root data/ilsvrc12 --output output --batch-images 10 --phase val --weight models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --split val --test-scales 320


