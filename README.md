The Pytorch Implementation of ''MASNet: A Robust Deep Marine Animal Segmentation Network''. 

## Introduction
In this project, we use Ubuntu 16.04.5, Python 3.7, Pytorch 1.7.1 and two NVIDIA RTX 2080Ti GPU. 

## Running

### Testing

Download the pretrained model [pre-trained model (baidu)](https://pan.baidu.com/s/11HCTFlHOgesSPCjsjINN2g (t4dg))(t4dg).

Check the model and image pathes in config.yaml and scripts/test.py, then run:

```
python test.py
```

### Training

To train the model, you need to first prepare our [dataset](https://drive.google.com).

Check the dataset path in config.yaml, and then run:
```
python train.py
```

## Citation

If you find MASNet is useful in your research, please cite our paper:

```

```






