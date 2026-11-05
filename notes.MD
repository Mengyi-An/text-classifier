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

4. 文件说明及函数记录

   1\) data.py:
      - split_dataset: 划分数据集
      - tokenize_dataset：convert text into tokens, and the type of tokens is a dictornary

   2\) baseline.py
      - 文本转向量：用 TF-IDF 把每篇文章变成一个 64042 维的向量，每一维对应词表里的一个词。
      - 训练：LogisticRegression 在训练集上 fit，学出每个词对每个类别的权重。
      - 预测和评估：模型对 val 和 test 做 predict 得到预测类别，再用 accuracy_score 和真实标签对比，算出准确率。

   3\) model.py
      - 主要是写了一个模型类，利用nn.Module继承的 pytorch 的 功能
      - 功能有：将 token 转成 vector，然后获得最后一层的 cls，再调用 classifier (参数为 cls)对类别进行预测。其中 classifier 中包含了 dropout 和 linear，注意在 classifier 中没有写 softmax，因为在计算 crossentropy中已经包含了 softmax 的计算。

   4\) train.py
      - get_dataloaders：将数据集一个 batch 一个 batch 的读取出来并转成 tensor
      - run_epoch: 计算 loss 和 acc，这里区分 model.train()和 model.eval()
      - save_metrics: 保存模型结果到 metrics.json
      - train: 整个模型训练的调度器，负责按顺序把seed, optimizer, training, test, save results这些函数串起来。

5. tokenizer 切分时，会在头尾自动加两个特殊 token：[CLS] 和 [SEP]，分别代表文本的开始和结束。它们和普通 token 一样，也会被转成 vector。

以下是一个简单的 example：

```
原文：      apple reports profit
tokenize：  [CLS] apple reports profit [SEP]
编号：       101   6207   ...    ...    102
```

**[CLS]**（classification，编号 101）
- 永远放在开头。
- 不代表任何词，是一个"汇总位"。
- 经过注意力后，它会收集整句话的信息，所以分类时用它的向量。

**[SEP]**（separator，编号 102）
- 放在结尾，表示"一段文本到这里结束"。
- BERT 预训练时，有时会同时输入两句话（比如判断第二句是不是第一句的下一句）。这时用 [SEP] 把它们分开：`[CLS] 句子A [SEP] 句子B [SEP]`。

6. dropout
Dropout 的作用是防止过拟合，也就是防止模型只会死记训练数据。nn.Dropout(0.2)，意思是 [CLS] 那 768 个数字里，每一步随机有大约 20% 被清零，剩下的数字会放大一点，让总量大致不变。下一步再重新随机选一批清零。到了验证和测试的时候，dropout 自动关掉，所有数字都用上。

