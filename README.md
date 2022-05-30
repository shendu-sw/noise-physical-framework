# Noise Physical Framework

This repository is the implementation for the paper "A CNN with Noise Inclined Module and Denoise Framework for Hyperspectral Image Classification" [[paper](https://arxiv.org/abs/2205.12459)].

## Dependencies

* Pytorch
* pytorch_metric_learning

## Data Preparation

* Pavia University data
  * PaviaU.mat: data file
  * 1\, 2\, 3\, 4\, 5\: five independent runs
    * TRLabel.mat: train label
    * TSLabel.mat: test label
* Houston2013 data

## Implementation

* Training

```python
python main.py --method NoiPhy --dataset pavia --neighbor 5
```

> - method: NoiPhy -- Proposed Method, 3DCNN -- 3D-CNN, SSFTTnet -- SSFTTNet, PResNet -- PResNet, HybridSN -- HybridSN
> - dataset: pavia or houston2013

* Testing

```python
python main.py --method NoiPhy --dataset pavia --neighbor 5 -e
```

## Acknowledge

* 3D-CNN: [A. B. Hamida, A. Benoit, P. Lambert, and C. B. Amar, “3-d deep learning approach for remote sensing image classification,” IEEE TGRS, vol. 56, no. 8, pp. 4420–4434, 2018.](https://ieeexplore.ieee.org/abstract/document/8344565)

* SSFTTNet: [L. Sun, G. Zhao, Y. Zheng, and Z. Wu, “Spectral-spatial feature tokenization transformer for hyperspectral image classification,” IEEE TGRS, 2022.](https://ieeexplore.ieee.org/abstract/document/9684381)
* HybridSN: [S. K. Roy, G. Krishna, S. R. Dubey, and B. B. Chaudhuri, “Hybridsn: Exploring 3-d-2-d cnn feature hierarchy for hyperspectral image classification,” IEEE GRSL, vol. 17, no. 2, pp. 277–281, 2019.](https://ieeexplore.ieee.org/abstract/document/8736016)
* PResNet: [M. E. Paoletti, J. M. Haut, R. Fernandez-Beltran, J. Plaza, A. J. Plaza, and F. Pla, “Deep pyramidal residual networks for spectralspatial hyperspectral image classification,” IEEE TGRS, vol. 57, no. 2, pp. 740–754, 2018.](https://ieeexplore.ieee.org/abstract/document/8445697)

## Citation

If you find this work helpful for your research, please consider citing:

    @article{gong2022,
        Author = {Zhiqiang Gong and Ping Zhong and Jiahao Qi and Panhe Hu},
        Title = {A CNN with Noise Inclined Module and Denoise Framework for Hyperspectral Image Classification},
        journal={arXiv Preprint arXiv: 2205.12459},
        Year = {2022}
    }