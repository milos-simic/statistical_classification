# How to Control the Error Rates of Binary Classifiers

This repository is the official implementation of the code to replicate the experiments from study "How to Control the Error Rates of Binary Classifiers". The paper presentes a theoretical framework for turning binary classifiers into statistical tests whose class-conditional error rates can be controlled and set in advance - but without retraining the original classifier.

The code for running the experiments and analyzing their results is in the form of Jupyter notebooks with comments.

Two experiments were originally run:
1. The first experiment shows how new normality tests can be derived from a neural network classifier.
2. The second experiment demonstrates how the binary SVM classifier developed for classifying loan applications as risky or non-risky can be turned into statistical tests.

## Requirements

The experiments were carried out using Python 3.9.4,and the results were analyzed with R 4.0.5.

### Python Requirements

The required packages are listed in file requirements.txt.

To install requirements all at once using pip, run the following command:

```setup
pip install -r requirements.txt
```

You can also install them one by one. Here is the list of the packages that were used in the original study:

| package | version |
| ------- | ------- |
| jupyter | 1.0.0   |
| numpy   | 1.20.2  |
| pandas  | 1.2.4   |
| rpy2    | 3.4.4   |
| scikit-learn | 0.24.1 |
| scipy | 1.6.2 | 
| matplotlib | 3.4.1 |
| joblib | 1.0.1 | 
| codecarbon | 1.2.0 |
| statsmodels | 0.12.2 |

### R Requirements

The following R libraries were used in the original study:

| package | version |
| ------- | ------- |
| nortest | 1.0-4   |
| ggplot2 | 3.3.3   |
| ggpubr  | 0.4.0.999 |
| boot    | 1.3-28 |
| binom   | 1.1-1  |
| sqldf   | 0.4-11 |
| latex2exp | 9.5.0 |
| RColorBrewer | 1.1.-2 |
| viridis | 0.6.1 |
| ggsci | 2.9 |
| IRkernel | 1.1.1 |

To install a library from the above list, e.g. nortest, run the following command:

```setup
install.packages('nortest')
```

Newer versions of the libraries should also work, but may requre the code for analysis to be modified.

#### A Note on Running R Notebooks

The analysis code is in Jupyter R notebooks, so make sure that you are able to run such notebooks. [This simple tutorial](https://developers.refinitiv.com/en/article-catalog/article/setup-jupyter-notebook-r) may prove helpful.

## Deriving Statistical Normality Tests from a Neural Network Classifier
An already trained neural network was taken from [Simić (2020)](https://arxiv.org/abs/2009.13831).

The normality tests are derived from it in notebook **nn_evaluate**. Simply open the notebook and follow the instructions to replicate the original experiment.

If you navigate to the directory where **nn_evaluate** is located and type the following command in your command line interface:
```setup
jupyter notebook
```
the jupyter server should start and you should be able to see a list of notebooks. Open **nn_evaluate** and start running its cells. If you are not familiar with jupyter, [this tutorial](https://realpython.com/jupyter-notebook-introduction/) will be useful.

The obtained results are analyzed in notebook **nn_analyze**.

### Pre-trained Models

The neural tests of normality and tests of loan application risk which we derived are not available at the moment, but you can very easily obtain your own versions by running **nn_evaluate**.

You can download the original neural network here:

- [Neural network for detecting normal distributions](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

