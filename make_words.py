import pandas as pd

def is_chars_present(your_word, your_chars):
    counter = 0
    #Removing duplicates
    your_word = list(set(your_word)) 
    your_chars = list(set(your_chars))
    for i in your_chars:
        if(i in your_word):
            counter += 1
    return counter


def make_words(letter_list, n = 5):
    word_sug_df = pd.DataFrame(pd.read_csv("Wordle.txt", sep="\t")["words"])
    word_sug_df["words"] = word_sug_df["words"].str.lower()

    #Filter for 5 letter words
    word_sug_df=word_sug_df[word_sug_df.words.apply(lambda x: len(str(x))==5)]

    word_sug_df["score"] = word_sug_df.apply(lambda x: is_chars_present(str(x.words),letter_list),axis =1)

    #score_of_word = []
    
    #for i in word_sug_df["words"]:
        #score_of_word.append(is_chars_present(letter_list, i))
    
    #word_sug_df["score"] = score_of_word

    word_sug_df.sort_values(by=["score"], inplace=True, ascending=[False])

    top_words = list(word_sug_df["words"][0:n])
    top_words_df = word_sug_df[word_sug_df.words.isin(top_words)]
    top_words_df.set_index("words", inplace=True)

    return top_words, top_words_df

        
    

    



    
