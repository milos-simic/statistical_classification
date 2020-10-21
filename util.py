import os
from bisect import bisect
import numpy as np
import pandas as pd
import scipy.stats
import random
import datetime
import pickle
import os
import pathlib

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector

from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.base import BaseEstimator, TransformerMixin

# Define the function to save an array of samples to a file
def save_to_file(data, filename):
    f = open(filename, "w")
    for row in data:
        line = ",".join([str(x) for x in row])
        f.write(line + "\n")
    f.close()

# Define a function to load a set of samples from a file
def load_from_file(filename):
    data = []
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines:
        xs = line.split(',')
        xs = [float(x) for x in xs]
        data.append(xs)
    f.close()
    return data

# For calculating empirical CDF
def ecdf(x, sample):
    sample.sort() # sample HAS TO BE SORTED in the ASCENDING (NON-DESCENDING) order
    i = bisect(sample, x)
    return float(i) / len(sample)


# For "inverting" empirical CDF
def pseudoInvEcdf(sample, p):
    n = len(sample)
    i = 0
    s = 1.0 / n
    while s < p and i < n:
        s = s + 1.0 / n
        i = i + 1
    if i == n:
        i = n - 1
    x = sample[i] #min([x for x in sample if ecdf(x, sample) >= p])
    return x

# Define a function which maps samples to descriptors (this is the feature engineering step)
def createDescriptor(sample, q):
    sample.sort()
    m = int(1 / q)
    n = len(sample)
    # The fimensions of the descriptors are sample size and the following descriptive statistics:
    maximum = max(sample)
    minimum = min(sample)
    mean = np.mean(sample)
    median = np.median(sample)
    sd = np.std(sample)
    #kurtosis = scipy.stats.kurtosis(sample)
    #skewness = scipy.stats.skew(sample)

    standardized_sample = [(x - mean) / sd for x in sample]
    
    # ... as well the selected quantiles
    descriptor = [pseudoInvEcdf(standardized_sample, j*q) for j in range(1, m+1)]
    descriptor = descriptor + [n, mean, sd, minimum, maximum, median]

    return descriptor

# For checking id xs contains only finite values
def onlyFinite(xs):
    for x in xs:
        if np.isinf(x) or np.isnan(x):
            return False
    return True

# This function assigns names to the descriptor features
def names(v):
    q = 10
    c = 10
    return ["p{}%".format(q*i) for i in range(1, c+1)] + ["n",  "mean", "sd", "max", "min", "med"]

# For the Lilliefors test
rstring = """
    function(sample){
        library(nortest)
        tryCatch({
            res <- lillie.test(sample)
            p <- res[[2]]
            return(p)
        }, error = function(e){
            return(0.0)
        })
    }
"""
lf=robjects.r(rstring)  

# Define the decision-making functions based on statistical tests: True - sample is normal, False - sample comes from
# a non-normal distribution

def lilliefors(sample, alpha):
    return lf(FloatVector(sample))[0] >= alpha

def shapiro_wilk(sample, alpha):
    return scipy.stats.shapiro(sample)[1] >= alpha

def anderson_darling(sample, alpha):
    if alpha == 0.01:
        c = 4
    elif alpha == 0.05:
        c = 2
    elif alpha == 0.1:
        c = 1
    result = scipy.stats.anderson(sample, dist='norm')
    statistic = result[0]
    critical_value = result[1][c]
    return statistic <= critical_value

def jarque_bera(sample, alpha):
    return scipy.stats.jarque_bera(sample)[1] >= alpha

# Create the access point for the tests based on their codes
def get_test(code):
    if code == 'SW':
        return shapiro_wilk, shapiro_wilk_statistic
    elif code == 'LF':
        return lilliefors, lilliefors_statistic
    elif code == 'AD':
        return anderson_darling, anderson_darling_statistic
    elif code == 'JB':
        return jarque_bera, jarque_bera_statistic

# This class is a wraper around tests
class TestClassifier(object):
    """docstring for Test"""
    def __init__(self, test, statistic, alpha, class_label=1, opposite_label=0):
        super(TestClassifier, self).__init__()
        self.test = test
        self.alpha = alpha
        self.statistic = statistic

        self._estimator_type = 'classifier'
        self.classes_ = [0, 1]

    def predict(self, samples):
        labels = [2 for sample in samples]
        for i, sample in enumerate(samples):
            try:
                is_normal = self.test(sample, self.alpha)
                if is_normal:
                    labels[i] = 1
                else:
                    labels[i] = 0
            except Exception as e: # due to numerical instability
                labels[i] = 2
        #labels = [1 if self.test(sample, self.alpha) else 0 for sample in samples]
        return labels

    def calculate_statistic(self, samples):
        if not all([type(x) == list for x in samples]):
            return self.statistic(samples)
        return np.array([self.statistic(sample) for sample in samples])

# This function creates a test-based classifier for given code and alpha
def get_standard_classifier(test_code, alpha):
    test, statistic = get_test(test_code)
    classifier = TestClassifier(test, statistic, alpha, 1, 0)
    return classifier

# Define the functions that calculate the statistics of normality tests

rstring_lf_stat = """
    function(sample){
        library(nortest)
        tryCatch({
            res <- lillie.test(sample)
            stat <- res[[1]]
            return(stat)
        }, error = function(e){
            return(-10.0)
        })
    }
"""
lf_stat=robjects.r(rstring_lf_stat)

def lilliefors_statistic(sample):
    return lf_stat(FloatVector(sample))[0]

def shapiro_wilk_statistic(sample):
    return scipy.stats.shapiro(sample)[0]

def anderson_darling_statistic(sample):
    return scipy.stats.anderson(sample, dist='norm')[0]

def jarque_bera_statistic(sample):
    return scipy.stats.jarque_bera(sample)[0]

# This function returns a randomly selected mean
def random_mean():
    return -100 + random.random() * 200

# This function returns a randomly selected standard deviation
def random_sigma():
    return 1 + random.random() * 19

# This function generates normal samples
def generate_normal_samples(ns, L):
    raw_samples = []
    for n in ns:
        for l in range(L):
            mu = random_mean()
            sigma = random_sigma()
            sample = np.random.normal(mu, sigma, n)
            raw_samples.append(sample.tolist())
        #print(n, len(raw_samples))
    return raw_samples

# Create the R string for the Pearson family. Make sure to have gsl and PearsonDS installed.
pearsonstring = """
function(n, m1, m2, m3, m4){
    library(gsl)
    library(PearsonDS)
    tryCatch({
        sample <- rpearson(n,moments=c(mean=m1,variance=m2,skewness=m3,kurtosis=m4))
        return(sample)
    }, error = function(e){
        return(FALSE)
    })
}
"""
generate_pearson=robjects.r(pearsonstring)

# Generate non-normal samples
def generate_pearson_nonnormal_samples(s_range, k_range, ns, L):
    # s_range is the set of skewness values to be supplied to generate_pearson
    # k_range is the set of kurtosis values to be supplied to generate_pearson
    h = 0
    raw_samples = []
    for n in ns:
        for s in s_range:
            for k in k_range:
                if k - s**2 - 1 >= 0 and not(s==0 and k==3):
                    h = h + 1
                    for l in range(L):
                        mean = random_mean()
                        sd = random_sigma()
                        response = generate_pearson(n, mean, sd, s, k)
                        if not(response[0] == False):
                            sample = response
                            #confs[(n, mean, sd, skewness, kurtosis)] = True
                            raw_samples.append(list(sample))
        #print(n, h, len(raw_samples))
    return raw_samples

# Define the function to assign the same label to all the samples
def label_samples(samples, label):
    return [list(sample) + [label] for sample in samples]

# Define the function for splitting the sets
def split(samples, train_size, labels=None, random_state=0):
    if labels is None:
        labels = [sample[-1] for sample in samples]

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, 
        stratify=labels, train_size=train_size, random_state=random_state)
        
    return X_train, X_test, y_train, y_test 

# This is deprecated.
def preprocess1(raw_samples, how={'method' : 'nothing'}):
    if how['method'] == 'nothing':
        return raw_samples

    elif how['method'] == 'descriptors':
        q = how['q']
        
        descriptors = [createDescriptor(sample[:-1], q) for sample in raw_samples]
        
        pruned = []
        
        for i in range(len(descriptors)):
            if onlyFinite(descriptors[i]):
                pruned += [descriptors[i] + [raw_samples[i][-1]]]
        
        return pruned

# Create a function for generating nicely formatted LaTeX tables
mean_sd_merge=lambda x: '${:.3f}\pm{:.3f}$'.format(x[0], x[1])

def get_latex_table(df, sort_by=None, merge_instructions=None, renamer=None, 
                    caption=None, label=None, float_format='$%.3f$', index=False):
    if sort_by is not None:
        df = df.sort_values(sort_by, axis='index')
    
    if merge_instructions is not None:
        for instruction in merge_instructions:
            merge_function = instruction['merge_function']
            new_column = instruction['new_column']
            columns_to_merge = instruction['columns_to_merge']
            df[new_column] = df[columns_to_merge].apply(merge_function, axis=1)
            df = df.drop(columns_to_merge, axis=1)
    
    if renamer is not None:
        df = df.rename(columns=renamer)
    
    latex = df.to_latex(index=index,
        float_format=float_format, escape=False, caption=caption, label=label)
    
    return latex

mean_sd_merge = lambda x: '${:.3f}\pm{:.3f}$'.format(x[0], x[1])

# Define the function for showing the current date and time in the selected format
def now(format='%Y-%m-%d %H-%M-%S.%f'):
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')

# Define the function that can calculate the following performance metrics: A - accuracy, TPR - true positive rate,
# PPV - positive predictive value, TNR - true negative rate, NPV - negative predictive value, F1 - the F1 measure,
# UR - the unresponsive rate (percentage of 2s in guesses that are caused by numerical instability)
def calculate_metrics(true_labels, guessed_labels, metrics=['A', 'TPR', 'PPV', 'TNR', 'NPV', 'F1', 'UR']):
    # Determine the elements of the confusion matrix
    matrix = sklearn.metrics.confusion_matrix(true_labels, guessed_labels, labels=[0, 1])
    
    T_nonnormal = matrix[0, 0]
    F_normal = matrix[0, 1]
    F_nonnormal = matrix[1, 0]
    T_normal = matrix[1, 1]
    
    # Determine the set specifics
    N_normal = sum(matrix[1, :])   # T_normal + F_nonnormal
    N_nonnormal = sum(matrix[0, :]) # T_nonnormal + F_normal
    N = N_normal + N_nonnormal
    
    scores = []
    
    # Calculate the chosen metrics
    if 'A' in metrics:
        score = (T_normal + T_nonnormal) / N
        scores.append(score)
    if 'TPR' in metrics:
        score = T_normal / N_normal
        scores.append(score)
    if 'PPV' in metrics:
        score = T_normal / (T_normal + F_normal)
        scores.append(score)
    if 'TNR' in metrics:
        score = T_nonnormal / N_nonnormal
        scores.append(score)
    if 'NPV' in metrics:
        score = T_nonnormal / (T_nonnormal + F_nonnormal)
        scores.append(score)
    if 'F1' in metrics:
        TPR = T_normal / N_normal
        PPV = T_normal / (T_normal + F_normal)
        score = 2 * TPR * PPV / (TPR + PPV)
        scores.append(score)
    if 'UR' in metrics:
        total = len(guessed_labels)
        undecided = len([x for x in guessed_labels if x == 2])
        score = undecided / total
        scores.append(score)
    
    return scores

# Define the function that evaliates a classifer's performance on a set of samples
def evaluate(samples, true_labels, classifier, metrics=['A', 'TPR', 'PPV', 'TNR', 'NPV', 'F1', 'AUROC', 'U'],
            n_range=None):
    # Guess the labels
    guessed_labels = classifier.predict(samples)

    # Calculate the performance metrics for the whole set
    total_scores = calculate_metrics(true_labels, guessed_labels, metrics=metrics)
    if 'AUROC' in metrics:
        prediction_scores = classifier.predict_proba(samples)[:, 1]
        where = np.where(~np.isnan(prediction_scores))[0]
        t = [true_labels[j] for j in where]
        ps = [prediction_scores[j] for j in where]
        auroc = sklearn.metrics.roc_auc_score(t, ps)
        i = metrics.index('AUROC')
        total_scores = total_scores[:i] + [auroc] + total_scores[i:]
    
    if n_range is None:
        return total_scores
    else:
        guessed_by_n = {n : [] for n in n_range}
        true_by_n = {n : [] for n in n_range}
        prediction_scores_by_n = {n : [] for n in n_range}

        for i, sample in enumerate(samples):
            n = len(sample)
            if n not in n_range:
                continue
            
            true_label = int(true_labels[i])
            true_by_n[n].append(true_label)

            guessed_label = int(guessed_labels[i])
            guessed_by_n[n].append(guessed_label)

            if 'AUROC' in metrics:
                prediction_score = prediction_scores[i]
                prediction_scores_by_n[n].append(prediction_score)

        all_scores = []
        for n in n_range:
            scores_for_n = calculate_metrics(true_by_n[n], guessed_by_n[n], metrics=metrics)
            if 'AUROC' in metrics:
                where = np.where(~np.isnan(prediction_scores_by_n[n]))[0]
                t = [true_by_n[n][j] for j in where]
                ps = [prediction_scores_by_n[n][j] for j in where]
                auroc_for_n = sklearn.metrics.roc_auc_score(t, ps)
                i = metrics.index('AUROC')
                scores_for_n = scores_for_n[:i] + [auroc_for_n] + scores_for_n[i:]
            all_scores.append(scores_for_n)
        
        return all_scores + [total_scores]

# Define a wrapper around evaluate that returns the results in the form of a Pandas dataframe
def evaluate_pretty(samples, true_labels, classifier, 
    metrics=['A', 'TPR', 'PPV', 'TNR', 'NPV', 'F1', 'AUROC'], n_range=None, index=None
    ):
    scores_lists = evaluate(samples, true_labels, classifier, metrics=metrics, n_range=n_range)

    columns = metrics

    if n_range is not None:
        for i in range(len(n_range)):
            scores_lists[i] = [n_range[i]] + scores_lists[i]
        scores_lists[-1] = ['overall'] + scores_lists[-1]

        columns = ['n'] + metrics

    if n_range is None:
        scores_lists = [scores_lists]

    results_df = pd.DataFrame(scores_lists, columns=columns)

    if index is not None:
        results_df = results_df.set_index(index, drop=True)

    return results_df


# https://stackoverflow.com/a/46730656/1518684
def get_activations(clf, X):
        hidden_layer_sizes = clf.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [X.shape[1]] + hidden_layer_sizes + \
            [clf.n_outputs_]
        activations = [X]
        for i in range(clf.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        clf._forward_pass(activations)
        return activations

# For separating samples by their labels
def separate_by_label(samples, labels):
    separated = {}
    for (sample, label) in zip(samples, labels):
        label = int(label)
        if label not in separated:
            separated[label] = []
        separated[label].append(sample)
    return separated

# For separating by size and, optinally, labels
def separate_by_size(samples, labels=None):
    separated_samples = {}
    if labels is not None:
        separated_labels = {}
    for i in range(len(samples)):
        sample = samples[i]
        n = len(sample)
        if n not in separated_samples:
            separated_samples[n] = []
        separated_samples[n].append(sample)
        if labels is not None:
            if n not in separated_labels:
                separated_labels[n] = []
            separated_labels[n].append(labels[i])

    if labels is None:
        return separated_samples
    else:
        return separated_samples, separated_labels

# For filtering the samples by label and size
def filter_samples(samples, labels, target_label=None, n=None):
    if target_label is None and n is None:
        return samples
    elif target_label is None and n is not None:
        for (sample, label) in zip(samples, labels):
            if len(sample) == n:
                filtered_samples.append(sample)
    elif target_label is not None and n is None:
        for (sample, label) in zip(samples, labels):
            if label == target_label:
                filtered_samples.append(sample)
    else:
        for (sample, label) in zip(samples, labels):
            if label == target_label:
                filtered_samples.append(sample)
                
    return filtered_samples

# Define the function that finds the lowest empirical quantile 
# that is greater than 100q% of the sample (0 < q <= 1).
# The sample has to be sorted in the ascending order in order for this function to work.

def get_quantile(sample, q):
    n = len(sample)
    i = 0
    s = 1.0 / n
    
    while s < q and i < n:
        s = s + 1.0 / n
        i = i + 1

    if i == n:
        i = n - 1

    return sample[i]

# Define the function to construct a sample's descriptor.
def get_descriptor(sample, q, sorted=True):
    # Make sure that the sample is sorted before calling this function
    if sorted == False:
        sample.sort()
        
    # Determine the descriptive statistics
    n = len(sample)
    maximum = max(sample)
    minimum = min(sample)
    mean = np.mean(sample)
    median = np.median(sample)
    sd = np.std(sample)
    #kurtosis = scipy.stats.kurtosis(sample, fisher=False)
    #skewness = scipy.stats.skew(sample)
    
    # Standardize the sample
    standardized_sample = [(x - mean) / sd for x in sample]
    
    # Determine the number of quantiles
    m = int(1 / q)
    
    # Determine the quantiles
    # Note: Optimize this step. All the quantiles can be found in one single pass.
    descriptor = [get_quantile(standardized_sample, j*q) for j in range(1, m + 1)]
    
    # Combine the quantiles and descriptive statistics to get the sample's descriptor
    descriptor = descriptor + [n, mean, sd, minimum, maximum, median]
    
    return descriptor

# Define a class to transform the samples to descriptors, that can be used in a sklearn pipeline
class DescriptorBuilder(TransformerMixin, BaseEstimator):
    def __init__(self, q=0.1):
        super(DescriptorBuilder, self).__init__()
        self.q = q
        
        # Set the names of the features in the descriptors
        self.features = ['q{:.2f}'.format(i * q) for i in range(1, int(1/q) + 1)]
        self.features += ['n', 'mean', 'sd', 'minimum', 'maximum', 'median']
        
    def fit(self, X, y=None):
        # Not needed, but present for compatibility.
        return self
    
    def transform(self, X, y=None):
        # Note: Currently works only on a list of lists or a single list.
        if isinstance(X, list):
            if all(isinstance(x, list) for x in X):
                X = [get_descriptor(x, q=self.q, sorted=False) for x in X]
                return pd.DataFrame(X)
            else:
                X = get_descriptor(X, q=self.q, sorted=False)
                return X
        else:
            # Pandas dataframes and numpy arrays are not supported for now.
            pass

# For completing all the steps in a pipe except for the last one (prediction)
# The neural net classifier consists of the descriptor builder, standard scaler, mean imputer, and the neural network.
def prepare_input(X, dbnn_pipe):
    X = dbnn_pipe['descriptor_builder'].transform(X)
    X = dbnn_pipe['scaler'].transform(X)
    X = dbnn_pipe['imputer'].transform(X)
    
    return X

# This functions traverses a dictionary to save all the figures in it to said directory
def traverse_and_save(dictionary, img_dir_path):
    if type(dictionary) is not dict:
        return
    for key in dictionary:
        if 'fig' in key:
            pathlib.Path(*img_dir_path.split(os.sep)).mkdir(parents=True, exist_ok=True)
            figure = dictionary[key]
            for extension in ['pdf', 'eps']:
                path = os.path.join(img_dir_path, img_dir_path.split(os.sep)[-1] + '_' + key) + '.' + extension 
                print('Saving', path)
                if 'savefig' in dir(figure):
                    figure.savefig(path, bbox_inches='tight')
                else:
                    figure.figure.savefig(path, bbox_inches='tight')
        else:
            traverse_and_save(dictionary[key], os.path.join(img_dir_path, key))