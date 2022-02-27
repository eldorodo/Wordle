#https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/
#https://towardsdatascience.com/clustering-algorithm-for-data-with-mixed-categorical-and-numerical-features-d4e3a48066a0
from distutils import dist
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull

from Factor2Approx import *
from Soving_using_Gurobi import *

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

import itertools
from itertools import combinations

from sklearn.mixture import GaussianMixture
import seaborn as sns

import multiprocessing
from multiprocessing import Pool

import random

def fuzzy_match_logic(your_word, choices):
    return process.extractOne(your_word, choices, scorer=fuzz.token_sort_ratio)[0]

# Hamming distance
def hammingDist(str1, str2):
    i = 0
    count = 0
 
    while(i < len(str1)):
        if(str1[i] != str2[i]):
            count += 1
        i += 1
    return count

def distance_of_strings(test_str1, test_str2):
    return len(test_str1) - len(set(test_str1).intersection(set(test_str2)))

def dist_matrix(word_list):
    dist_matrix_df = pd.DataFrame(0, index=word_list, columns=word_list)
    for i in range(len(word_list)):
        for j in range(len(word_list)):
            f = 0
            f+= distance_of_strings(word_list[i],word_list[j])
            f+= hammingDist(word_list[i],word_list[j])
            dist_matrix_df.loc[word_list[i],word_list[j]] = f

    return dist_matrix_df



def is_char_present(your_word, your_char):
    if(your_char in your_word):
        return 1
    else:
        return 0
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
def nearest_index(samples, your_array, nearby = 2):
    neigh = NearestNeighbors(n_neighbors=nearby, radius=0.4)
    neigh.fit(samples)
    indices_list = np.array(neigh.kneighbors(your_array, nearby, return_distance=False)[0]).tolist()
    return indices_list

def k_means_graph(X,k_max = 15):
    wcss = []  #Within-Cluster Sum of Square
    for i in range(1, k_max): 
        kmeans_model = KMeans(n_clusters = i, init = 'k-means++', random_state = 42, n_init = 5, max_iter = 50)
        kmeans_model.fit(X) 
        wcss.append(kmeans_model.inertia_)

    plt.plot(range(1,k_max), wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') 
    plt.show()

def missing_letters(shortlist_list):
    letter_list = []
    for i in shortlist_list:
        temp = [char for char in i]
        letter_list.extend(temp)
    set_difference = set(alphabet_list) - set(letter_list)

    return set_difference  



def shortlist_five_words(iter_words):
    max = 0
    final_combo = iter_words[0]
    for i in iter_words:
        word_combo = list(i)
        for combo in combinations(word_combo, 5):  # 2 for pairs, 3 for triplets, etc
            #print(combo)
            dist_df = dist_matrix(list(combo))
            curr_dist = dist_df.to_numpy().sum()
            if(curr_dist > max):
                max = curr_dist
                final_combo = combo
    return final_combo

def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def shortlist_words(wc):
    word_lists = []
    for combo in combinations(wc, 5):  # 2 for pairs, 3 for triplets, etc
        #print(combo)
        if(np.random.random(size=1)[0] <0.85):
            continue
        temp = []
        temp.extend(combo)
        dist_df = dist_matrix(combo)
        curr_dist = dist_df.to_numpy().sum()/2
        temp.append(curr_dist)
        word_lists.append(temp)
    return word_lists



#################################################################################

df = pd.DataFrame(pd.read_csv("Wordle.txt", sep="\t")["words"])
df["words"] = df["words"].str.lower()

#Filter for 5 letter words
df=df[df.words.apply(lambda x: len(str(x))==5)]

#Finding unique letters and filtering for only letters that have all five unique letters
df["unique_letters"] = df.words.apply(lambda x: ''.join(set(x)))
df=df[df.unique_letters.apply(lambda x: len(str(x))==5)]
del df["unique_letters"]

df.index = range(df.shape[0])


alphabet_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
  "r", "s", "t","u", "v", "w", "x", "y", "z"]

alphabet_counter = []

#Finding alphabet probability
for i in alphabet_list:
    counter = 0
    for j in df.words:
        if(is_char_present(j, i) == 0):
            counter += 1 #Counting if not present
        else:
            counter += 0    
        
    alphabet_counter.append(counter)
    
alphabet_counter = list(minmax_scale(np.array(alphabet_counter)))
num_bins = 26
bin_vals = list(np.linspace(1, 26, num=num_bins))
alphabet_counter = list(pd.qcut(pd.Series(alphabet_counter), 26, labels=bin_vals))

alphabet_embeddings = dict(zip(alphabet_list, alphabet_counter))

print("alphabet_embeddings",alphabet_embeddings)
    
#Adding columns
for i in range(5):
    col_name = 'letter'+ str(i+1)
    df[col_name] = df.apply(lambda x: str(x.words)[i],axis =1)
    #df[col_name].astype('category')

#df = pd.get_dummies(df, columns = ['letter1', 'letter2', 'letter3', 'letter4', 'letter5'])


#Encoding letters based on position
five_letters = ["letter1", "letter2", "letter3", "letter4", "letter5",]
for i in five_letters:
    values = df[i].value_counts(dropna=False).keys().tolist()
    counts = df[i].value_counts(dropna=False).tolist()
    counts = [1- c/df.shape[0] for c in counts]
    counts = list(minmax_scale(np.array(counts)))
    value_dict = dict(zip(values, counts))
    df[i].replace(value_dict,inplace=True)
    num_bins = 6
    #bin_vals = list(np.linspace(1, 26, num=num_bins))
    df[i] = df[[i]].apply(lambda x:pd.qcut(x, num_bins, labels=False, duplicates = 'drop'), axis = 0)
    #High, Medium, Low

#df["std_dev"] = df[['letter1', 'letter2', 'letter3', 'letter4', 'letter5']].std(axis=1)
#df["mean"] = df[['letter1', 'letter2', 'letter3', 'letter4', 'letter5']].mean(axis=1)

# df[alphabet_list] = -1

# for i in alphabet_list:
#     #new_col = str(i) + "_" + "pos"
#     df[i] = df.apply(lambda x: is_char_present(str(x.words),i) * alphabet_embeddings[i],axis =1)
#     #df[new_col] = df.apply(lambda x: str(x.words).find(i),axis =1)

df.set_index('words', inplace=True)
df.drop_duplicates(inplace=True)
print(df.head(20))


X = df.values
#print(X)
index_words = df.index

#k_means_graph(X,k_max=50)

def shortlist_of_words(X, k, x_index, nearby = 1):
    shortlist_words = []
    shortlist_words_cluster_wise =[]

    kmeans_model = KMeans(n_clusters = k, init = 'k-means++', random_state = 42, n_init = 100, max_iter = 1000)
    kmeans_model.fit(X)
    label_arr = np.array(kmeans_model.labels_)
    clus_centers =  np.array(kmeans_model.cluster_centers_)
    for i in clus_centers:
        near_indexes = nearest_index(X, [i], nearby=nearby)
        temp =[]
        for j in  near_indexes:
            temp.append(x_index[j])
            shortlist_words.append(x_index[j])
        shortlist_words_cluster_wise.append(temp)

    return shortlist_words, shortlist_words_cluster_wise

first_shortlist_words, first_shortlist_words_cl = shortlist_of_words(X, 6, index_words, 1)
print("first_shortlist_words", first_shortlist_words)

first_shortlist_dist = dist_matrix(first_shortlist_words)
first_shortlist_words_cl_iter = [list(x) for x in np.array(np.meshgrid(*first_shortlist_words_cl)).T.reshape(-1,len(first_shortlist_words_cl))]

shortlist_df = df[df.index.isin(first_shortlist_words)]
index_words_updated = shortlist_df.index
X2 = shortlist_df.values

second_shortlist_words = shortlist_five_words(first_shortlist_words_cl_iter)

print("second_shortlist_words", second_shortlist_words)

shortlist_df = df[df.index.isin(second_shortlist_words)]
index_words_updated = shortlist_df.index


print("missing letters first", missing_letters(first_shortlist_words))
print("missing letters second", missing_letters(second_shortlist_words))



######################### GMM #####################

n_components = np.arange(1, 1)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
#plt.show()

n_components = 38
gmm = GaussianMixture(n_components=n_components, n_init= 10, max_iter= 400)
gmm.fit(X)
prob = gmm.predict_proba(X)
labels = gmm.predict(X)

prob_df = pd.DataFrame(prob)

maxValues = prob_df.max(axis = 1)

gmm_df = pd.DataFrame(labels)
gmm_df.columns = ["labels"]
gmm_df["maxValues"] = maxValues
gmm_df["words"] = index_words

print(gmm_df.head())

#https://stackoverflow.com/questions/15705630/get-the-rows-which-have-the-max-value-in-groups-using-groupby
idx = gmm_df.groupby(['labels'])['maxValues'].transform(max) == gmm_df['maxValues']

new_gmm_df = gmm_df[idx]

final_combo_list_master =[]
uniq =[]
for i in range(10):
    stratified_df = stratified_sample_df(new_gmm_df, col = "labels", n_samples= 1)
    print(i, stratified_df)
    word_combo = list(stratified_df["words"])
    dist_df_all = dist_matrix(word_combo)
    t = combinations(word_combo, 4)
    lst_t = list(t)
    lst_t = random.sample(lst_t, int(len(lst_t)*1))
    for c in lst_t:
        d = tuple(sorted(c))
        if(d in uniq):
            continue
        uniq.append(d)
        temp = list(c)
        dist_df = dist_df_all[dist_df_all.index.isin(list(c))]
        dist_df = dist_df_all[list(c)]
        curr_dist = dist_df.to_numpy().sum()/2
        temp.append(curr_dist)
        final_combo_list_master.append(temp)
    

#print(final_combo_list_master)


final_combo = pd.DataFrame(final_combo_list_master, columns =['word1', 'word2', 'word3', 'word4', 'dist'])
#final_combo = pd.DataFrame(final_combo_list, columns =['word1', 'word2', 'word3', 'dist'])


df_list = final_combo.values.tolist()

ml = []
mls_join = []
for i in df_list:
    record = i[:-1]
    mls = missing_letters(record)
    mls_join.append(' '.join(list(mls)))
    ml.append(len(mls))

final_combo["num_missing_letters"] = ml
final_combo["missing_letters"] = mls_join


final_combo.sort_values(by=["num_missing_letters", 'dist'], inplace=True, ascending=[True,False])

print(final_combo.head(20))

final_combo.to_csv("best_words_wordle_4.csv",index=False)














