import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pickle.load(open("criterion.pkl", "rb"))

summaries = df.summary.values

st.title('Criterion Film Search')
user_input = st.text_input("Search here with keywords relating to themes, characters,"
                           " directors, country, year, plot... ")


def critsearch(text_input):
    # add the user's query to the film summaries
    array_with_query = np.insert(summaries, 0, user_input)

    # vectorize all of the docs using a sparse matrix with TF-IDF
    tfidf = TfidfVectorizer().fit_transform(array_with_query)

    # use dot product to find most similar vectors/summaries
    cosine_similarities = linear_kernel(tfidf[0], tfidf).flatten()

    # find the top 40, using -40 since they are reverse sorted
    related_docs_indices = cosine_similarities.argsort()[:-40:-1]

    # print the results with streamlit
    result_no = 1
    for i in related_docs_indices[1:]:
        st.write(result_no)
        summ = array_with_query[i].split('•')
        try:
            st.write('Title: ', df.iloc[i - 1].title, '\n', summ[0], '\n', summ[1])
            st.write(summ[2].split(maxsplit=1)[0])
            st.write(summ[2].split(maxsplit=1)[1])
        except:
            st.write(array_with_query[i])
        st.write('\n')
        result_no += 1


if user_input != '':
    critsearch(user_input)





