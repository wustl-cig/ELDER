# [Deep Equilibrium Learning of Explicit Regularizers for Imaging Inverse Problems](https://arxiv.org/abs/2303.05386)

There has been significant recent interest in the use of deep learning for regularizing imaging inverse problems. Most work in the area has focused on regularization imposed implicitly by convolutional neural networks (CNNs) pre-trained for image reconstruction. In this work, we follow an alternative line of work based on learning explicit regularization functionals that promote preferred solutions. We develop the Explicit Learned Deep Equilibrium Regularizer (ELDER) method for learning explicit regularizers that minimize a mean-squared error (MSE) metric. ELDER is based on a regularization functional parameterized by a CNN and a deep equilibrium learning (DEQ) method for training the functional to be MSE-optimal at the fixed points of the reconstruction algorithm. The explicit regularizer enables ELDER to directly inherit fundamental convergence results from optimization theory. On the other hand, DEQ training enables ELDER to improve over existing explicit regularizers without prohibitive memory complexity during training. We use ELDER to train several approaches to parameterizing explicit regularizers and test their performance on three distinct imaging inverse problems. Our results show that ELDER can greatly improve the quality of explicit regularizers compared to existing methods, and show that learning explicit regularizers does not compromise performance relative to methods based on implicit regularization.



## Requirements

run the

```
conda env create -f environment.yml
```

## Models

The pre-trained models can be downloaded from [Google drive](https://drive.google.com/drive/folders/1Q1DTyWffT6dGEaLMO3qa2l4U5QVaNVeG?usp=sharing).
Once downloaded, place them into `./ckpts`.
## How to run the code

ELDER:

```
python tune.py data_fidality=sr_test_x3 regularization=elder
```

DEQ:

```
python tune.py data_fidality=sr_test_x3 regularization=deq
```

### Citation
If you find the paper useful in your research, please cite the paper:
```BibTex
@ARTICLE{Zou.etal2023,
  author={Z. {Zou} and J. {Liu} and B. {Wohlberg} and U. S. {Kamilov}},
  journal={arXiv preprint arXiv:2303.05386},
  title={Deep Equilibrium Learning of Explicit Regularizers for Imaging Inverse Problems}, 
  year={2023},
}
