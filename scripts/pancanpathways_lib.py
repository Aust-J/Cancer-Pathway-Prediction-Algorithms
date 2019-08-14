#!/usr/bin/env python3
"""
Module containing helper functions for recommender system analysis of PanCanAtlas pathways data,
including implementation of Surprise package and edge swap shuffling, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import collections
import statistics
import random

from surprise import Reader
from surprise import Dataset
from surprise import prediction_algorithms
from surprise.model_selection import cross_validate

sns.set(style='ticks', palette='Set2')


def num_differences_btwn_dfs(df1, df2):
    """ credit to 
    https://stackoverflow.com/questions/44265983/getting-number-of-differences-between-2-dataframes-in-python
    """
    return df1.count().sum() - (df1 == df2).astype(int).sum().sum()


def random_binary_df(df):
    """
    Returns dataframe with dimensions the same as input df but filled with random 0s and 1s
    """
    random = df.copy()
    for i in range(len(random.index)):
        for j in range(len(random.columns)):
            random.iat[i,j]=np.random.randint(0,2)
    return(random)


def gam_matrix_to_surprise_format(gam_matrix):
    """
    Convert genomic alteration matrix format (samples by pathways) to format required for 
    input to Surprise library (each line in the format sample\tpathway\tvalue)
    """
    gam_surpriseFormat = pd.melt((gam_matrix.reset_index()), id_vars=["SAMPLE_BARCODE"])
    gam_surpriseFormat.columns = ["sample", "pathway", "value"]
    return gam_surpriseFormat


def get_edges_list(df):
    """
    Input: binary pandas df
    Returns: list of unique edges (cells with value=1) identified by index/column labels
             each edge is a list of length 2
    """
    edges = []
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            if df.iat[i,j]==1:
                new_edge = [df.index[i], df.columns[j]]
                if new_edge not in edges:
                    edges.append(new_edge)
    print("Created list of {} unique edges.".format(len(edges)))
    return(edges)


def remove_pathways_with_nulls(gam_pathwayLevel, all_pathways):
    """
    Remove pathways with nulls
    """
    print("\n### Null value check and filtering: pathways ###")
    print("------------------------------------------------")

    # check if any pathways have NAs for sample alteration status
    pathways_with_nulls = gam_pathwayLevel.columns[gam_pathwayLevel.isnull().any()].tolist()
    print("Pathways containing NaNs: ", pathways_with_nulls); print("")

    # for each pathway with nulls, check how many of the samples are null
    for pathway in pathways_with_nulls:
            num_null_samples = len(gam_pathwayLevel[gam_pathwayLevel[pathway].isnull()].index)
            print("There are {} samples with value NaN in the '{}' pathway.\n".format(
                    num_null_samples, pathway))   

    pathways_noNulls = all_pathways.copy()
    gam_pathwayLevel_noNullPathways = gam_pathwayLevel.copy()

    for pathway in pathways_with_nulls:
            pathways_noNulls.remove(pathway)
            gam_pathwayLevel_noNullPathways.drop(pathway, axis=1, inplace=True)
            print("Removed '{}' pathway from because it contained NaNs.\n".format(pathway))

    assert pathways_noNulls == list(gam_pathwayLevel_noNullPathways.columns), "Error while removing null pathways from GAM matrix, pathway list"

    print("Remaining pathways: {}\n".format(list(gam_pathwayLevel_noNullPathways.columns)))
    print("Matrix is now {} samples by {} pathways.".format(gam_pathwayLevel_noNullPathways.shape[0], 
            gam_pathwayLevel_noNullPathways.shape[1]))
    print("------------------------------------------------\n")
    
    return(gam_pathwayLevel_noNullPathways)


def remove_samples_with_nulls(gam_pathwayLevel, all_samples):
    """
    Remove samples with nulls
    """
    print("\n### Null value check and filtering: samples ###")
    print("-----------------------------------------------")
    # check if any samples have NAs for sample alteration status
    samples_with_nulls = gam_pathwayLevel.index[gam_pathwayLevel.isnull().any(axis=1)].tolist()
    print("Number of samples containing nulls: ", len(samples_with_nulls)); print("")

    samples_noNulls = all_samples.copy()
    gam_pathwayLevel_noNullSamples = gam_pathwayLevel.copy()

    for sample in samples_with_nulls:
            samples_noNulls.remove(sample)
            gam_pathwayLevel_noNullSamples.drop(sample, axis=0, inplace=True)
    print("Removed all samples containing nulls.\n")

    assert samples_noNulls == list(gam_pathwayLevel_noNullSamples.index), "Error while removing null pathways from GAM matrix, pathway list"

    print("Matrix is now {} samples by {} pathways.".format(gam_pathwayLevel_noNullSamples.shape[0], 
            gam_pathwayLevel_noNullSamples.shape[1]))

    print("-----------------------------------------------\n")

    return(gam_pathwayLevel_noNullSamples)



def plot_sample_freq_dist(df):
    """
    """
    # dict containing sample: [freq of zeros, freq of ones]
    sample_to_alterationFreq_dict = collections.defaultdict(list)
    for sample in df.index:
            freqs = df.loc[sample].value_counts(normalize=True, dropna=True)
            if len(list(freqs.items())) == 2:
                    # if the frequencies are both nonzero, the frequency series 
                    # will contain values for both categories (ones and zeros)
                    sample_to_alterationFreq_dict[sample].extend([freqs[0], freqs[1]])
                    # adds freq of zeros, freq of 1s to list

            elif len(list(freqs.items())) == 1:
                    # this sample is either all 0s or all 1s, so the freqs series
                    # will drop the missing category. need to handle diferently.
                    category, freq = list(freqs.items())[0]
                    if category == 0.0: # the entire sample was 0s, so we want to add (1,0)
                            sample_to_alterationFreq_dict[sample].extend([freq, freq-1])
                    elif category == 1.0: #the entire sample was 1s, so we ant to add (0,1)
                            sample_to_alterationFreq_dict[sample].extend([freq-1, freq])
                    else:
                            raise Exception("Something went wrong making sample to alteration freq dict, inner loop")
            else:
                    raise Exception("Something went wrong making sample to alteration freq dict, outer loop")


    samples_zero_freqs = []
    samples_one_freqs = []

    for zero_freq, one_freq in sample_to_alterationFreq_dict.values():
        samples_zero_freqs.append(zero_freq)
        samples_one_freqs.append(one_freq)
        
    print("Stats for distribution of zeros (not altered) across all samples/pathways: \n", stats.describe(samples_zero_freqs))
    print("Stats for distribution of ones (altered) across all samples/pathways: \n", stats.describe(samples_one_freqs))
    print("")

    fig, ax = plt.subplots()
    sns.distplot(samples_one_freqs, ax=ax, kde=False, color="indianred", label="Altered")
    sns.distplot(samples_zero_freqs, ax=ax, kde=False, color="steelblue", label="Not altered")

    plt.legend(bbox_to_anchor=(1.01,.9))
    plt.title("Sample alteration frequency across all pathways")
    plt.xlabel("alteration frequency")
    plt.ylabel("number of samples")


def plot_pathway_freq_dist(df):
    """
    """
    for_barplot = df.apply(lambda x: pd.value_counts(x), axis=0).transpose()
    for_barplot.columns=["0", "1"]

    totals = [zeros+ones for zeros,ones in zip(for_barplot["0"], for_barplot["1"])]
    zeros = [zeros/total for zeros,total in zip(for_barplot["0"], totals)]
    ones = [ones/total for ones, total in zip(for_barplot["1"], totals)]

    # reorder pathways from highest freq of alterations to lowest
    for_barplot_temp = for_barplot.copy()
    for_barplot_temp["ones_freq"] = ones
    pathway_order = list(for_barplot_temp["ones_freq"].sort_values(ascending=False).index)
    
    for_barplot_sorted = for_barplot_temp.reindex(labels=pathway_order).drop("ones_freq", axis=1)

    totals_sorted = [zeros+ones for zeros,ones in zip(for_barplot_sorted["0"], for_barplot_sorted["1"])]
    zeros_sorted = [zeros/total for zeros,total in zip(for_barplot_sorted["0"], totals_sorted)]
    ones_sorted = [ones/total for ones, total in zip(for_barplot_sorted["1"], totals_sorted)]

    names = list(for_barplot_sorted.index)
    # alter some pathway names just for aesthetics on plot
    # comment out for now -- don't want to hard-code in case want to use this 
    # when the pathways might be in a different order (e.g. for undergrads)
    #names[0]="Cell Cycle"
    #names[1]="RTK/RAS"
    
    inds = [x for x,_ in enumerate(for_barplot_sorted.index)]
    barWidth= 0.6

    plt.rcParams['figure.figsize']=(9.5,5)
    plt.rcParams.update({"axes.titlesize": "x-large"})

    p1 = plt.bar(inds, zeros_sorted, label="Not altered", width=barWidth, color="steelblue", alpha=.7)
    p2 = plt.bar(inds, ones_sorted, bottom=zeros_sorted, label="Altered", width=barWidth, color="indianred", alpha=.7)

    plt.xticks(inds, names)
    plt.xlabel("pathway")
    plt.ylabel("relative frequency")
    plt.title("Pathway alteration frequencies across all samples")

    plt.legend((p2[0], p1[0]), ("Altered", "Not altered"), bbox_to_anchor=(1.01,.9))

    values = [round(x,2) for x in ones_sorted]
    for i in range(len(df.columns)):
        plt.text(x=inds[i]-.23, y=.96, s=format(values[i],".2f"), color="white")

    ## uncomment to display text version of frequencies per pathway
    # for pathway in df.columns:
    #         freqs = df[pathway].value_counts(normalize=True, dropna=True)
    #         print("{}: \n 0: {} \n 1 {}\n".format(pathway, freqs[0], freqs[1]))
    

def run_surprise_algo_crossval(algoName, genomic_alteration_matrix, numCrossValFolds=10, outputMeasures=["RMSE","MAE"], similarityMetric='cosine'):

        if similarityMetric not in ["cosine", "msd", "pearson", "pearson_baseline"]:
                raise Exception("Invalid similarity metric. Similarity metric must be 'cosine', 'msd', 'pearson', or 'pearson_baseline'")
        
        # only call algorithm with similarity measures option for appropriate algorithms
        if algoName in ["KNNBasic", "KNNWithMeans", "KNNWithZScore", "KNNBaseline"]:
                sim_opts_dict = {"name": similarityMetric}
                algo = getattr(prediction_algorithms, algoName)(sim_options=sim_opts_dict)
        else:
                algo = getattr(prediction_algorithms, algoName)()

        print("\nConverting genomic alteration matrix to Surprise input format... ", end="")
        gam_for_surprise = gam_matrix_to_surprise_format(genomic_alteration_matrix)
        print("done.\n")
        print("Surprise input matrix has {} rows in the format `sample    pathway    value`\n".format(gam_for_surprise.shape[0]))

        # define Surprise Reader object 
        # pathway activation values are either 0 or 1, so specify in rating scale
        reader = Reader(rating_scale=(0,1))

        # create Surprise data object using reader
        print("Creating Surprise data object... ", end="")
        dataObj = Dataset.load_from_df(df=gam_for_surprise, reader=reader)
        print("done.\n")

        # run Surprise algorithm k-fold cross validation
        print("Running {}-fold cross validation for {} algorithm...\n".format(numCrossValFolds, algoName))
        crossval_output_dict = cross_validate(algo, dataObj, cv=numCrossValFolds, measures=outputMeasures, verbose=True)
        print("")

        return(crossval_output_dict)



def create_shuffled_df_final_march2019_clean(edges_list, df):
    """
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make an empty df that will be the shuffled df
    df_shuffled = pd.DataFrame().reindex_like(df)
    
    num_swaps = 0
    num_reinserts = 0
    num_loops = 0
    
    list_of_newly_created_edges = []
    
    while len(edges_shuffled)>440: # came up with this cutoff after trial and errror
        
        if (num_loops%1000==0):
            print("Loop number: ", num_loops)
            print("Length of edges_shuffled list: ", len(edges_shuffled))
            print("Number of edges so far: ", df_shuffled.sum().sum())
            
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists. only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]

        if (swapped_edge1 not in list_of_newly_created_edges) and (swapped_edge2 not in list_of_newly_created_edges):

            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1

            list_of_newly_created_edges.append(swapped_edge1)
            list_of_newly_created_edges.append(swapped_edge2)

            num_swaps += 1

        else:
            # put one of the edges back on top of the list and move the other to a random place
            edges_shuffled.insert(0, swapped_edge1)

            edges_shuffled.insert(random.randint(0, len(edges_shuffled)), swapped_edge2)

            num_reinserts += 1

                
        num_loops += 1

    
    print("Swapped {} edge pairs.".format(num_swaps))
    print("New number of edges: {}".format(int(df_shuffled.sum().sum())))

    print("Number of reinserts: {}".format(num_reinserts))
    print("Number of iterations through while loop: {}".format(num_loops))

    
    return(df_shuffled)



# same as below, but slightly cleaned up and removed some commented-out parts. 
# sent to undergrads, I think. 
def create_shuffled_df_20190317(edges_list, df):
    """
    same as post-henry one, but with removed commented-out parts
    and filled in NaNs with zeros 
    I think this is the one I sent to the undergrads (not tha they used it...)
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make an empty df that will be the shuffled df
    df_shuffled = pd.DataFrame().reindex_like(df)
    
    num_swaps = 0
    num_reinserts = 0
    num_loops = 0
    
    list_of_newly_created_edges = []
    
    while len(edges_shuffled)>440: # came up with this cutoff after trial and errror
        
        if (num_loops%1000==0):
            print("Loop number: ", num_loops)
            print("Length of edges_shuffled list: ", len(edges_shuffled))
            print("Number of edges so far: ", df_shuffled.sum().sum())
            
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        #if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
        # henry pointed out that we shouldn't care whether we are remaking edges in the original data.
        # in fact if we don't, the fake data will be anticorrelated with the real data which isn't great either. 
            
        if (swapped_edge1 not in list_of_newly_created_edges) and (swapped_edge2 not in list_of_newly_created_edges):

            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1

            list_of_newly_created_edges.append(swapped_edge1)
            list_of_newly_created_edges.append(swapped_edge2)


            num_swaps += 1

        else:#if (swapped_edge1 in list_of_newly_created_edges) or (swapped_edge2 in list_of_newly_created_edges):

            # put one of the edges back on top of the list and move the other to a random place
            edges_shuffled.insert(0, swapped_edge1)

            edges_shuffled.insert(random.randint(0, len(edges_shuffled)), swapped_edge2)

            num_reinserts += 1

        #else:
            #raise Exception("Something went wrong during shuffling and new edge creation.")
                
        num_loops += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    #how is it possible that the number of edges changes? aren't i deleting two and adding two each time?
    print("New number of edges: {}".format(int(df_shuffled.sum().sum())))
    
    print("Number of reinserts: {}".format(num_reinserts))
    print("Number of iterations through while loop: {}".format(num_loops))

    return(df_shuffled.fillna(0))


# FIXES AFTER DISCUSSION WITH HENRY
# Remove condition to not make new edge if it was in the original data,
# because this would result in the new matrix being negatively correlated with the old.
# (two versions of of this: with and without deleting "old" edges)
def create_shuffled_df_20190301_henry_second(edges_list, df):
    """
        same as below, but comment out lines that delete old edges 
        ( #df_shuffled.at[edge1[0], edge1[1]] = 0
          #df_shuffled.at[edge2[0], edge2[1]] = 0 )
        Output is still short about 500 edges, but this makes sense because it's the exact number left on the list! 
        Still need to decide what to do with these. possibly fill with zero.
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make an empty df that will be the shuffled df
    df_shuffled = pd.DataFrame().reindex_like(df)
    
    num_swaps = 0
    num_reinserts = 0
    num_loops = 0
    
    list_of_newly_created_edges = []
    
    while len(edges_shuffled)>440: # came up with this cutoff after trial and errror
        
        if (num_loops%1000==0):
            print("Loop number: ", num_loops)
            print("Length of edges_shuffled list: ", len(edges_shuffled))
            print("Number of edges so far: ", df_shuffled.sum().sum())
            
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        #if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
        # henry pointed out that we shouldn't care whether we are remaking edges in the original data.
        # in fact if we don't, the fake data will be anticorrelated with the real data which isn't great either. 
            
        if (swapped_edge1 not in list_of_newly_created_edges) and (swapped_edge2 not in list_of_newly_created_edges):

            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1

            list_of_newly_created_edges.append(swapped_edge1)
            list_of_newly_created_edges.append(swapped_edge2)

            # delete old edges
            # NEED TO CHECK THIS -- did it actually make a difference? fills in 0's on what would otherwise be NaNs
            # yes -- results in significantly fewer edges if commented out. WHY?!?
            #df_shuffled.at[edge1[0], edge1[1]] = 0
            #df_shuffled.at[edge2[0], edge2[1]] = 0

            num_swaps += 1

        else:#if (swapped_edge1 in list_of_newly_created_edges) or (swapped_edge2 in list_of_newly_created_edges):

            # put one of the edges back on top of the list and move the other to a random place
            edges_shuffled.insert(0, swapped_edge1)

            edges_shuffled.insert(random.randint(0, len(edges_shuffled)), swapped_edge2)

            num_reinserts += 1

        #else:
            #raise Exception("Something went wrong during shuffling and new edge creation.")
                
        num_loops += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    #how is it possible that the number of edges changes? aren't i deleting two and adding two each time?
    print("New number of edges: {}".format(int(df_shuffled.sum().sum())))

    
    print("Number of reinserts: {}".format(num_reinserts))
    print("Number of iterations through while loop: {}".format(num_loops))

    
    return(df_shuffled)

def create_shuffled_df_20190301_henry_first(edges_list, df):
    """
    # Henry: If I never create any edges that were also in the original data, the new matrix will end up 
        # being negatively correlated with the old. Take out this requirement
        # todo (fixes from Josh meeting 20190222)
        # 1. start with a blank matrix, not a copy, to make shuffled df
        # 2. before swapping, check to see if either new edge already exists in *shuffled* matrix. 
        #    If so, don't proceed because you will end up creating only 1 new edge instead of 2, decreasing overall # edges
        #    Instead, if encountered, put the edges back in the stack randomly (or at least one edge) and try again later
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make an empty df that will be the shuffled df
    df_shuffled = pd.DataFrame().reindex_like(df)
    
    num_swaps = 0
    num_reinserts = 0
    num_loops = 0
    
    list_of_newly_created_edges = []
    
    while len(edges_shuffled)>440: # came up with this cutoff after trial and errror
        
        if (num_loops%1000==0):
            print("Loop number: ", num_loops)
            print("Length of edges_shuffled list: ", len(edges_shuffled))
            print("Number of edges so far: ", df_shuffled.sum().sum())
            
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        #if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
        # henry pointed out that we shouldn't care whether we are remaking edges in the original data.
        # in fact if we don't, the fake data will be anticorrelated with the real data which isn't great either. 
            
        if (swapped_edge1 not in list_of_newly_created_edges) and (swapped_edge2 not in list_of_newly_created_edges):

            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1

            list_of_newly_created_edges.append(swapped_edge1)
            list_of_newly_created_edges.append(swapped_edge2)

            # delete old edges
            # NEED TO CHECK THIS -- did it actually make a difference? fills in 0's on what would otherwise be NaNs
            # yes -- results in significantly fewer total number of edges
            df_shuffled.at[edge1[0], edge1[1]] = 0
            df_shuffled.at[edge2[0], edge2[1]] = 0

            num_swaps += 1

        else:#if (swapped_edge1 in list_of_newly_created_edges) or (swapped_edge2 in list_of_newly_created_edges):

            # put one of the edges back on top of the list and move the other to a random place
            edges_shuffled.insert(0, swapped_edge1)

            edges_shuffled.insert(random.randint(0, len(edges_shuffled)), swapped_edge2)

            num_reinserts += 1

        #else:
            #raise Exception("Something went wrong during shuffling and new edge creation.")
                
        num_loops += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    #how is it possible that the number of edges changes? aren't i deleting two and adding two each time?
    print("New number of edges: {}".format(int(df_shuffled.sum().sum())))

    
    print("Number of reinserts: {}".format(num_reinserts))
    print("Number of iterations through while loop: {}".format(num_loops))

    
    return(df_shuffled)



# FIXES AFTER JOSH MEETING 20190222 ---
# 1) start with blank matrix
# 2) edge existence check in shuffled matrix (to avoid decreased # edges at end)
def create_shuffled_df_20190301(edges_list, df):
    """
        # to do (fixes from Josh meeting 20190222)
        # 1. start with a blank matrix, not a copy, to make shuffled df
        # 2. before swapping, check to see if either new edge already exists in *shuffled* matrix. 
        #   If so, don't proceed because you will end up creating only 1 new edge instead of 2, decreasing overall # edges
        #   Instead, if encountered, put the edges back in the stack randomly (or at least one edge) and try again later
        
        # --> I don't think this ended up working... moved on quickly to Henry's suggestions
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make an empty df that will be the shuffled df
    df_shuffled = pd.DataFrame().reindex_like(df)
    
    num_swaps = 0
    num_reinserts = 0
    num_loops = 0
    
    list_of_newly_created_edges = []
    
    while len(edges_shuffled)>1:
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
        #print("Length of shuffled edges list: {} {}".format(len(edges_shuffled), len(list_of_newly_created_edges)))
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
            
            
            if (swapped_edge1 not in list_of_newly_created_edges) and (swapped_edge2 not in list_of_newly_created_edges):
            
                df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
                df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1
            
                list_of_newly_created_edges.append(swapped_edge1)
                list_of_newly_created_edges.append(swapped_edge2)
                                                                                        
                # delete old edges
                df_shuffled.at[edge1[0], edge1[1]] = 0
                df_shuffled.at[edge2[0], edge2[1]] = 0
            
                num_swaps += 1
                
            elif (swapped_edge1 in list_of_newly_created_edges) or (swapped_edge2 in list_of_newly_created_edges):
                
                # put one of the edges back on top of the list and move the other to a random place
                edges_shuffled.insert(0, swapped_edge1)
                
                edges_shuffled.insert(random.randint(0, len(edges_shuffled)), swapped_edge2)
                
                num_reinserts += 1
            
            else:
                raise Exception("Something went wrong during shuffling and new edge creation.")
                
        num_loops += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    #how is it possible that the number of edges changes? aren't i deleting two and adding two each time?
    print("New number of edges: {}".format(int(df_shuffled.sum().sum())))

    
    print("Number of reinserts: {}".format(num_reinserts))
    print("Number of iterations through while loop: {}".format(num_loops))

    
    return(df_shuffled)

def create_shuffled_df_20190227(edges_list, df):
    """
        # to do (fixes from Josh meeting 20190222)
        # 1. start with a blank matrix, not a copy, to make shuffled df
        # 2. 
        # note: 
        reults in a matrix with many NaN values. 
        # this is obviously a problem. I don't know what Josh wanted me to do here. 
        # we wanted to start with a blank matrix instead of a copy so that the shuffled df
        # would be more dissimilar from the real one. but what should the NaNs be? 
        # ---> (later)I ended up filling in the NaNs with 0's when I went to use it. 
        # ---> I think that makes sense but not sure. 
        # at this point: 
        #  still need to do the fix that will stop number of edges from decreasing...
        # if this makes the NaN problem a lot smaller, maybe there rest should just be zeros? 
        # maybe need to ask josh. 
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make an empty df that will be the shuffled df
    df_shuffled = pd.DataFrame().reindex_like(df)
    
    num_swaps = 0
    while len(edges_shuffled)>1:
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1
            # delete old edges
            df_shuffled.at[edge1[0], edge1[1]] = 0
            df_shuffled.at[edge2[0], edge2[1]] = 0
            
            num_swaps += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    #how is it possible that the number of edges changes? aren't i deleting two and adding two each time?
    print("New number of edges: {}".format(int(df_shuffled.sum().sum())))

    
    return(df_shuffled)



# FIX AFTER LAB MEETING -- forgot to delete old edges
def create_shuffled_df_20190224(edges_list, df):
    """
    fixex after lab meeting -- realized I forgot to delete old edges during shuffle/swap
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make a copy of the df for shuffling
    df_shuffled = df.copy()
    
    num_swaps = 0
    while len(edges_shuffled)>1:
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1
            # delete old edges
            df_shuffled.at[edge1[0], edge1[1]] = 0
            df_shuffled.at[edge2[0], edge2[1]] = 0
            
            num_swaps += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    #how is it possible that the number of edges changes? aren't i deleting two and adding two each time?
    print("New number of edges: {}".format(df_shuffled.sum().sum()))

    
    return(df_shuffled)

def create_shuffled_df_20190224_recursive(edges_list, df, num_swaps=0, stopping_num_swaps=6500):
    """
    """
    print("Initial number of edges: {}".format(len(edges_list)))

    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make a copy of the df for shuffling
    df_shuffled = df.copy()
    
    #num_swaps = 0
    while len(edges_shuffled)>1:
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1
            # delete old edges
            df_shuffled.at[edge1[0], edge1[1]] = 0
            df_shuffled.at[edge2[0], edge2[1]] = 0
            num_swaps += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    num_diffs = num_differences_btwn_dfs(G, df_shuffled)
    print("Number of differences from original matrix: {}".format(num_diffs))
    
    if num_diffs>=20000:
        return(df_shuffled)
    else:
        return(create_shuffled_df_20190224_recursive(get_edges_list(df_shuffled), df_shuffled, num_swaps))



# FIRST ATTEMPTS
def create_shuffled_df_firstattempt(edges_list, df):
    """
    ## OLD/ORIGINAL VERSION -- DOES NOT REMOVE OLD EDGES AFTER CREATING SWAPPED EDGES! BAD!

    """
    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make a copy of the df for shuffling
    df_shuffled = df.copy()
    
    num_swaps = 0
    while len(edges_shuffled)>1:
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1
            num_swaps += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    return(df_shuffled)

def create_shuffled_df_firstattempt_recursive(edges_list, df, num_swaps=0, stopping_num_swaps=6500):
    """
    """
    # shuffle list of edges (random.shuffle method is in place)
    edges_shuffled = edges_list.copy()
    random.shuffle(edges_shuffled)
    
    # make a copy of the df for shuffling
    df_shuffled = df.copy()
    
    #num_swaps = 0
    while len(edges_shuffled)>1:
        edge1 = edges_shuffled.pop()
        edge2 = edges_shuffled.pop()
    
        # check if the swapped edges exists
        # only make the swap if neither exits
        swapped_edge1 = [edge1[0], edge2[1]]
        swapped_edge2 = [edge2[0], edge1[1]]
    
        if (swapped_edge1 not in edges_list) and (swapped_edge2 not in edges_list):
            df_shuffled.at[swapped_edge1[0], swapped_edge1[1]] = 1
            df_shuffled.at[swapped_edge2[0], swapped_edge2[1]] = 1
            num_swaps += 1

    print("Swapped {} edge pairs.".format(num_swaps))
    
    if num_swaps>=10000:
        return(df_shuffled)
    else:
        return(create_shuffled_df_firstattempt_recursive(get_edges_list(df_shuffled), df_shuffled, num_swaps))
