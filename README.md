The Pytorch Implementation of ''MASNet: A Robust Deep Marine Animal Segmentation Network''. 

## Introduction
In this project, we use Ubuntu 16.04.5, Python 3.7, Pytorch 1.7.1 and two NVIDIA RTX 2080Ti GPU. 

## Running

### Testing

Download the pretrained model [pre-trained model](https://drive.google.com/file/d/1SKRIxUnG1GEA5h1mfSf2YbpPiFBmNaD8/view?usp=share_link).

Check the model and image pathes in config.yaml and scripts/test.py, then run:

```
python test.py
```

### Training

To train the model, you need to first prepare our [dataset](https://drive.google.com/file/d/1RNP_zJgbJeY5ibEcVfYQMgxMBjbfkT0B/view?usp=share_link).

Check the dataset path in config.yaml, and then run:
```
python train.py
```

## Citation

If you find MASNet is useful in your research, please cite our paper:

```

```






