import pandas as pd
import streamlit as st
from make_words import *

st.title('Wordle Helper')
st.subheader("The best Five words that tries most of alphabets are BINGO, VELUM, HOWDY, CRAFT and SPIKE !!!")

alphabet_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
  "r", "s", "t","u", "v", "w", "x", "y", "z"]

options = st.sidebar.multiselect(
     'Choose the letters',
     alphabet_list,
     default = "a")

#st.sidebar.write('You selected:', options)

num_words = list(range(1,15))
n = int(st.sidebar.selectbox("Number of suggestions required",
                     num_words))

option_list = list(options)
# Create a button, that when clicked, shows a text
if(st.sidebar.button(f"Suggest")):
    top_words, top_words_df = make_words(option_list, n)
    st.table(top_words_df)





    
    
