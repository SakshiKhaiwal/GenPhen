# Genotype to phenotype prediction of natural variation in S. cerevisiae.

In this project, we investigate the associations between the phenotypes and genotypes in a natural yeast population. We have gathered >200 phenotypes from various studies for 1011 cerevisiae strains (ref. paper). To construct meaningful models that can provide predictions of the phenotype in a novel population based on its genotypes, we explored several linear and nonlinear machine learning (ML) methods. Since we are interested in the prediction of quantitative phenotypes, we utilized regression-based methods, such as ridge regression, support vector regression (SVR), gradient-boosted machines (GBM), and deep neural networks (DNN). While ridge regression and SVR (depending on the type of kernel used) can identify a linear relationship between the input features and the target variable, GBMs and DNNs can map more complex nonlinear relationships between the two. We built a convenient and flexible ML pipeline implementing the four methods and consisting of different steps required for constructing these genotype-phenotype maps, such as data pre-processing, feature selection, and model learning using two types of hyperparameter optimization techniques, namely random and Bayesian optimization.


# Installations
```
- git clone git@github.com:SakshiKhaiwal/GenPhen.git
- cd pipeline
- conda create -n GenPhen python=3.8
- conda install --file requirements.txt
```

# Running the pipeline



