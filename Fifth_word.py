import pandas as pd
from make_words import *

def Convert(string):
    li = list(string.split(" "))
    return li

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


alphabet_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
  "r", "s", "t","u", "v", "w", "x", "y", "z"]

def missing_letters(shortlist_list):
    letter_list = []
    for i in shortlist_list:
        temp = [char for char in i]
        letter_list.extend(temp)
    set_difference = set(alphabet_list) - set(letter_list)
  
    return set_difference

  
best_4_df = pd.read_csv("best_words_wordle_4.csv")
best_df_top = best_4_df.head(20)

fifth_word_list = []
dist_list = []
mls_list = []
num_mls_list = []

for i in range(best_df_top.shape[0]):
  curr_words = []
  curr_words.append(best_df_top.loc[i,"word1"])
  curr_words.append(best_df_top.loc[i,"word2"])
  curr_words.append(best_df_top.loc[i,"word3"])
  curr_words.append(best_df_top.loc[i,"word4"])
  letters = best_df_top.loc[i,"missing_letters"]
  letter_list = Convert(letters)
  num_sug = 5
  suggested_words, suggested_df = make_words(letter_list, num_sug)
  max_dist = 0
  for j in suggested_words:
    curr_words.append(j)
    print(curr_words)
    dist_df = dist_matrix(curr_words)
    curr_dist = dist_df.to_numpy().sum()/2
    if(curr_dist > max_dist):
      max_dist = curr_dist
      fifth_word = j
      mls = missing_letters(curr_words)
      num_mls = len(mls)
      mls_join = ' '.join(list(mls))
      dist1 = curr_dist
    curr_words.pop()
  fifth_word_list.append(fifth_word)
  dist_list.append(dist1)
  mls_list.append(mls_join)
  num_mls_list.append(num_mls)

del best_df_top["dist"]
del best_df_top["num_missing_letters"]
del best_df_top["missing_letters"]
best_df_top["word5"] = fifth_word_list
best_df_top["dist"] = dist_list
best_df_top["num_missing_letters"] = num_mls_list
best_df_top["missing_letters"] = mls_list

best_df_top.reset_index(inplace=True)

best_df_top.sort_values(by=["num_missing_letters", 'dist'], inplace=True, ascending=[True,False])


best_df_top.to_csv("best_words_4plus1.csv",index=False)





