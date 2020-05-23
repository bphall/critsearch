import pickle
import pandas as pd

df = pickle.load(open( "criterion.pkl", "rb"))


# instead of doc2vec, use a td-idf classic search function here to pull from df.summary

