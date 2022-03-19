# Feature-Distilled Transformer for UAV Tracking
### Changhong Fu, Haobo Zuo, Guangze Zheng, Junjie Ye, and Bowen Li
## Abstract
Occlusion, fast motion, and illumination variation
are prone to cause object feature pollution, which lead to crucial problems for visual tracking, especially from unmanned aerial vehicle (UAV) perspectives in the intelligent transportation field. The key reason is that most trackers directly exploit search patch features for location estimation without considering if they are contaminated. To address this issue, this work proposes an efficient and effective feature-distilled Transformer (FDT) for aerial tracking, which can alleviate feature
pollution by purifying search features with uncontaminated temporal information. FDT consists
of two primary parts, a feature encoder to enhance internal attention of feature maps and a distillation decoder guided by the purifying strategy. Specifically, the distillation decoder fully exploits temporal information about feature pollution from the last frame, thereby efficiently guiding the distillation of feature maps. Consequently, pure features of the object are accurately reserved to improve the tracking performance in challenging UAV scenarios. Exhaustive experiments on four authoritative UAV tracking benchmarks have validated that FDT achieves the state-of-the-art performance, especially on sequences with severe feature pollution. In addition, FDT has strongly proved its practicability with 32.7 frames per second in the real-world tests on an aerial platform.

![Workflow of our tracker](https://github.com/vision4robotics/FDT-tracker/blob/main/images/workflow.jpg)
## About Code
### 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2. Please install related libraries before running this code:

      pip install -r requirements.txt
### 2. Test
Download pretrained model: [FDTmodel](https://pan.baidu.com/s/1fTM66ZzcCQjPGg1_2-bDiA)(code:r0fy) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit.git) to set test_dataset.

       python test.py 
	        --dataset UAV123                #dataset_name
	        --snapshot snapshot/FDTmodel.pth  # tracker_name
	
The testing result will be saved in the `results/dataset_name/tracker_name` directory.
### 3. Train
#### Prepare training datasets

Download the datasets:

[VID](https://image-net.org/challenges/LSVRC/2017/)
 
[COCO](https://cocodataset.org/#home)

[GOT-10K](http://got-10k.aitestunion.com/downloads)

[LaSOT](http://vision.cs.stonybrook.edu/~lasot/)

#### Train a model

To train the FDT model, run `train.py` with the desired configs:

       cd tools
       python train.py

### 4. Evaluation
We provide the tracking [results](https://pan.baidu.com/s/1PoKNWFKJ40Loeu_E1GJuPQ)(code: 35rw) of UAV123@10fps, DTB70, UAVTrack112_L, and UAV123. If you want to evaluate the tracker, please put those results into `results` directory.

        python eval.py 	                          \
	         --tracker_path ./results          \ # result path
	         --dataset UAV123                  \ # dataset_name
	         --tracker_prefix 'FDTmodel'   # tracker_name
### 5. Contact
If you have any questions, please contact me.

Haobo Zuo

Email: <1951684@tongji.edu.cn>
## Demo Video
[![Watch the video](https://i.ytimg.com/vi/X-Js1hNL5JY/maxresdefault.jpg)](https://youtu.be/X-Js1hNL5JY)
## Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot.git). We would like to express our sincere thanks to the contributors.
