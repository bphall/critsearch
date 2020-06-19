The purpose of this mini-project was to implement a search engine for content on the Criterion Channel, since its platform was simpler than Netflix for the purpose of data collection. 

The actual engine is fairly simple, and uses TD-IDF in sklearn to produce sparse matrix representations of each film's summary. Thus, this engine promotes results bases on exact word (though not phrase, such as bigram or trigram) matches, as opposed to connotation matches that use word embeddings, or deep-learned models such as Google's BERT. 
