#!/usr/bin/env python3

"""

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import collections
import statistics

sns.set()

from surprise import Reader
from surprise import Dataset
from surprise import prediction_algorithms
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

from pancanpathways_lib import *

# read in disease labels
disease_from_tumormap = pd.read_table("./disease_fromTumorMap.txt", header=0, index_col=None)
sampleID_to_disease = disease_from_tumormap.set_index("samples")["disease"].to_dict()


# read data into df. 
# this is the "pathway-level genomic alteration matrix (gam)" from mmc4.xlsx of the paper
gam_pathwayLevel_mixedtypes = pd.read_table("/Users/aweinstein/Dropbox/UCSC/stuartlab/Recommender_on_pancan_pathway/Data/genomic_alteration_matrices.Pathway_level.tsv", 
                                        header=0, index_col=0)

gam_pathwayLevel = gam_pathwayLevel_mixedtypes.astype('float64') # make sure all columns are floats not ints

print("\nLoaded pathway-level genomic alteration matrix.\n")

initial_num_samples = gam_pathwayLevel.shape[0]
initial_num_pathways = gam_pathwayLevel.shape[1]
all_pathways = list(gam_pathwayLevel.columns)
all_samples = list(gam_pathwayLevel.index)

print("Matrix is {} samples by {} pathways.\n".format(initial_num_samples, initial_num_pathways))
#sns.heatmap(gam_pathwayLevel)

print("Plotting sample and pathway frequency distributions (raw data)... ", end="")
# plot_sample_freq_dist(gam_pathwayLevel)
# plt.savefig("sample_freq_dist_all.pdf", bbox_inches="tight")
# plt.close()

# plot_pathway_freq_dist(gam_pathwayLevel)
# plt.savefig("pathway_freq_dist_all.pdf", bbox_inches="tight")
# plt.close()
print("done\n")

# Make version of matrix with all pathways containing null vaules removed
gam_pathwayLevel_noNullPathways = remove_pathways_with_nulls(gam_pathwayLevel, all_pathways)


# Make version of matrix with all samples containing null vaules removed
gam_pathwayLevel_noNullSamples = remove_samples_with_nulls(gam_pathwayLevel, all_samples)

print("Plotting sample and pathway frequency distributions (data w/ samples containing null values removed)... ", end="")
# plot_sample_freq_dist(gam_pathwayLevel_noNullSamples)
# plt.savefig("sample_freq_dist_noNullSamples.pdf", bbox_inches="tight")
# plt.close()

# plot_pathway_freq_dist(gam_pathwayLevel_noNullSamples)
# plt.savefig("pathway_freq_dist_noNullSamples.pdf", bbox_inches="tight")
# plt.close()
print("done\n")

print("Plotting sample and pathway frequency distributions (data w/ pathways containing null values removed)... ", end="")
# plot_sample_freq_dist(gam_pathwayLevel_noNullPathways)
# plt.savefig("sample_freq_dist_noNullPathways.pdf", bbox_inches="tight")
# plt.close()

# plot_pathway_freq_dist(gam_pathwayLevel_noNullPathways)
# plt.savefig("pathway_freq_dist_noNullPathways.pdf", bbox_inches="tight")
# plt.close()
print("done\n")

# run on real data
run_surprise_algo_crossval("SlopeOne", gam_pathwayLevel_noNullSamples, numCrossValFolds=5)


# run on random binary data
print("\n###### Random binary matrix ######")
G_random = random_binary_df(gam_pathwayLevel_noNullSamples)
run_surprise_algo_crossval("SlopeOne", G_random, numCrossValFolds=5)




###############################
# Example usage for non-automated crossval 
print("\n~~~~~~~~~~~~~~~~~~~~~~~~")
print("\n######### Examples for non-automated cross validation #########")

print("\nConverting genomic alteration matrix to Surprise input format... ", end="")
gam_for_surprise = gam_matrix_to_surprise_format(gam_pathwayLevel_noNullSamples)
print("done.\n")
print("Surprise input matrix has {} rows in the format `sample    pathway    value`\n".format(gam_for_surprise.shape[0]))

# define Surprise Reader object 
# pathway activation values are either 0 or 1, so specify in rating scale
reader = Reader(rating_scale=(0,1))

# create Surprise data object using reader
print("Creating Surprise data object... ", end="")
dataObj = Dataset.load_from_df(df=gam_for_surprise, reader=reader)
print("done.\n")


# train-test split and the fit() method 
# (see https://surprise.readthedocs.io/en/stable/getting_started.html#cross-validate-example)
print("\n### Example usage for train-test split and the fit() method ###\n")

algo = getattr(prediction_algorithms, "SlopeOne")()

trainset, testset = train_test_split(dataObj, test_size=.20)
algo.fit(trainset)

predictions=algo.test(testset)

accuracy.rmse(predictions)
accuracy.mae(predictions)

# Train on a whole testset and use the predict() method
# (see https://surprise.readthedocs.io/en/stable/getting_started.html#cross-validate-example)
print("\n### Example usage for training on a whole testset and using the predict() method ###\n")

algo = getattr(prediction_algorithms, "SlopeOne")()

trainset = dataObj.build_full_trainset()
algo.fit(trainset)

sample = "TCGA-OR-A5J1-01"
for pathway in all_pathways:
        algo.predict(sample, pathway, verbose=True)

print("\nTrue values:")
print(gam_pathwayLevel_noNullSamples.loc[sample])
print("")


###############################
# Example using built-in dataset
# see https://surprise.readthedocs.io/en/stable/getting_started.html#cross-validate-example
print("\n~~~~~~~~~~~~~~~~~~~~~~~~")
print("\n######### Example using built-in movielens-100k dataset #########\n")

movieset_data = Dataset.load_builtin('ml-100k')
algo = getattr(prediction_algorithms, "SVD")()
cross_validate(algo, movieset_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")
