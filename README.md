# Explainable Learning with Gaussian Processes
In this repo, we provide MATLAB code to reproduce the results from our paper [Explainable Learning with Gaussian Processes](https://arxiv.org/abs/2403.07072v1), which is currently available on the [arXiv](https://arxiv.org/abs/2403.07072v1).

> **Abstract:** The field of explainable artificial intelligence (XAI) attempts to develop
methods that provide insight into how complicated machine learning methods make
predictions. Many methods of explanation have focused on the concept of feature
attribution, a decomposition of the model's prediction into individual
contributions corresponding to each input feature. In this work, we explore the
problem of feature attribution in the context of Gaussian process regression
(GPR). We take a principled approach to defining attributions under model
uncertainty, extending the existing literature. We show that although GPR is a
highly flexible and non-parametric approach, we can derive interpretable,
closed-form expressions for the feature attributions. When using integrated
gradients as an attribution method, we show that the attributions of a GPR
model also follow a Gaussian process distribution, which quantifies the
uncertainty in attribution arising from uncertainty in the model. We
demonstrate, both through theory and experimentation, the versatility and
robustness of this approach. We also show that, when applicable, the exact
expressions for GPR attributions are both more accurate and less
computationally expensive than the approximations currently used in practice.


## Instructions
To generate all figures (as .png files), you just need to run `main.m`. The code should run with no issues using Matlab 2022a or later. All generated figures and tables will be saved to the results folder. 
```
git clone https://github.com/KurtButler/2024_attributions_paper
```

## Data Availability
In our experiments, we used several publicly available data sets from the UCI Machine Learning Repository:
- [Breast Cancer Wisconsin (Prognostic)](https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic), from William Wolberg, W. Street, and Olvi Mangasarian
- [New Taipei City Housing Data](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set), from I-Cheng Yeh at the Department of Civil Engineering, Tamkang University
- [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), from Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis


## Citation
If you use any code or results from this project in your academic work, please cite our paper:
```
@article{butler2024explainable,
      title={Explainable Learning with Gaussian Processes}, 
      author={Kurt Butler and Guanchao Feng and Petar M. Djuric},
      year={2024},
      eprint={2403.07072},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

