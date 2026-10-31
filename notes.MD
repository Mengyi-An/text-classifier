## traning set and test set

1. fit_transform(train_set)

   1\) fit: learn the vocabulary and compute idf values
   2\) transform: convert text into vector

2. transform(test_set)

3. how to compute tf, df, idf

   1\) TF (term frequency)
   tf(w, m) = the number of times word w appears in document m.(当前文档m 中 w 出现的次数)

   2\) DF (document frequency)
   df(w) = the number of documents that contain w at least once.(在多少个文档中出现过)

   3\) IDF (inverse document frequency)
   idf(w) = log(M / df(w))
   where M is the total number of documents.

   4\) TF-IDF
   tf-idf(w, m) = tf(w, m) × idf(w)
