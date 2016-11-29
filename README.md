# Deep Primal-Dual Networks

## Datasets
The training data used for the depth only super-resolution method is available [here](https://drive.google.com/drive/folders/0B-TLBHhAnzw8VWc3SWpHaVd1N0U?usp=sharing).
It consists of three tar files, one for each benchmark dataset used in our paper:
  - dataset_mb.tar: noiseless Middleburry
  - dataset_nmb.tar: noisy Middleburry
  - dataset_tm_ta.tar: ToFMark
  
They mainly differ in the scaling of the depth values.
  
The training data for the guided depth super-resolution is available [here](https://drive.google.com/drive/folders/0B-TLBHhAnzw8V0xzT3VxclhmWm8?usp=sharing).

## Publications
The papers explaining the methods are on arxiv:
- [ATGV-Net: Accurate Depth Super-Resolution](https://arxiv.org/abs/1607.07988)
- [A Deep Primal-Dual Network for Guided Depth Super-Resolution](https://arxiv.org/abs/1607.08569)

If you find the code, or the data useful for your research, please cite

```
@inproceedings{riegler16dsr,
  title={ATGV-Net: Accurate Depth Super-Resolution},
  author={Riegler, Gernot and R\"{u}ther, Matthias and Bischof Horst},
  booktitle={European Conference on Computer Vision},
  year={2016}
}
```

```
@inproceedings{riegler16gdsr,
  title={A Deep Primal-Dual Network for Guided Depth Super-Resolution},
  author={Riegler, Gernot and Ferstl, David and R\"{u}ther, Matthias and Bischof Horst},
  booktitle={British Machine Vision Conference},
  year={2016}
}
```
