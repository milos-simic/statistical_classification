# How to Control the Error Rates of Binary Classifiers

This repository is the official implementation of the code to replicate the experiments from paper "How to Control the Error Rates of Binary Classifiers". 

The paper presents a theoretical framework for **turning binary classifiers into statistical tests** whose class-conditional **error rates can be set in advance**: without retraining the original classifier!

The code for running the experiments and analyzing their results is in the form of Jupyter notebooks with comments. If you are not familiar with Jupyter, you can read [this short tutorial](https://realpython.com/jupyter-notebook-introduction/).

Two experiments were originally run:
1. The first experiment shows how two **new normality tests** can be derived from a neural network classifier.
2. The second experiment demonstrates how the binary **SVM classifier** developed for classifying loan applications as risky or non-risky can be **turned into two statistical tests**.

## Requirements

The experiments were carried out using Python 3.9.4 and the results were analyzed with R 4.0.5.

The following python and R packages were used in the original experiment:

| python package | version | R package | version |
| ------- | ------- | ----  | ---- |
| jupyter | 1.0.0   | nortest | 1.0-4   |
| numpy   | 1.20.2  | ggplot2 | 3.3.3   |
| pandas  | 1.2.4   | ggpubr  | 0.4.0.999 |
| rpy2    | 3.4.4   | boot    | 1.3-28 |
| scikit-learn | 0.24.1 | binom | 1.1-1  |
| scipy | 1.6.2 | sqldf   | 0.4-11 |
| matplotlib | 3.4.1 | latex2exp | 9.5.0 |
| joblib | 1.0.1 | RColorBrewer | 1.1.-2 |
| codecarbon | 1.2.0 | viridis | 0.6.1 |
| statsmodels | 0.12.2 | ggsci | 2.9 |
|             |        |IRkernel | 1.1.1 |

### Python Requirements

The required packages are listed in file requirements.txt that is provided in this repository.

To install requirements all at once using pip, run the following command:

```setup
pip install -r requirements.txt
```

You can also install them one by one. 

### R Requirements

To install a library from the above list, e.g. nortest, run the following command:

```setup
install.packages('nortest')
```

Newer versions of the libraries should also work, but may requre the code for analysis to be modified.

#### A Note on Running R Notebooks

The analysis code is in Jupyter R notebooks, so make sure that you are able to run such notebooks. [This simple tutorial](https://developers.refinitiv.com/en/article-catalog/article/setup-jupyter-notebook-r) may prove helpful.

## Deriving Statistical Normality Tests from a Neural Network Classifier

### Data

The settings are specific here because one can create as much data as needed (or wanted). To create the datasets for this experiment, run notebook **nn_prepare_data**. 

If you navigate to the directory where **nn_prepare_data** is located and type the following command in your command line interface:

```setup
jupyter notebook
```
the jupyter server should start and you should be able to see a list of notebooks. Open **nn_prepare_data** and follow the instructions there.

### The Original Classifier

An already trained neural network was taken from [Simić (2020)](https://arxiv.org/abs/2009.13831). You can find it in file **dbnn1.p** located in folder **classifiers** of this repository.

### Derive and Evaluate the Tests

The normality tests are derived in notebook **nn_evaluate**. Open **nn_evaluate** and follow the instructions there to conduct the experiment.

### Analyze the Results

The results obtained in **nn_evaluate** are analyzed in notebook **nn_analyze**. Open and run it to analyze the results.

## Deriving Statistical Tests of Loan Application Risk

### Data

The data (HMEQ data) can be obtained from [here](https://www.kaggle.com/ajay1735/hmeq-data).

### The Original Classifier

To train an SVM binary classifier, run notebook **hmeq_evaluate**. 

### Derive and Evaluate the Tests

The tests are derived and evaluated in notebook **hmeq_evaluate**. Simply follow the instructions to conduct the experiment.

### Analyze the Results

The results obtained in **hmeq_evaluate** can be analyzed in notebook **hmeq_analyze**. 

