# BMRC-ACOS
Use BMRC to complete Aspect Sentiment Quadruple Extraction tasks in ACOS(rest,laptop), QUAD(rest15,rest16) and ASQE dataset.

# Contributions
- 我们首先提出了一种基于 MRC 用于情感四重提取任务的神经模型。 它也是第一个通过设计语言相关模板而适用于中英文评论数据集的ACOS提取模型。
- 我们的模型能够通过将其转换为显式问题来更好地解决方面或观点术语的隐含情况。 此外，优化了匹配方面和意见项的配对策略。
- 我们的模型在英文评论数据集 ASQP（包括两个数据集 rest15 和 rest16）和中文评论数据集 Chinese-ASQE 上都实现了 SOTA 性能。

**首先针对QUAD源数据我们发现在转化为下标的时候会有下面这几条数据中label无法与句子中的词匹配到，我们进行了修正**
  ```python
  15res train 689 '/waitress' -> 'waitress'
  15res test 408 'Maitre-D-' -> 'Maitre-D'
  
  16res train 867 '/waitress' -> 'waitress'
  16res dev 92 'Maitre-D-' -> 'Maitre-D'
  16res dev 93 '·rudeness' -> 'rudeness'
  16res test 260 "'service" -> "service"
  
  # 同时我们还发现一些错误数据
  15res dev 118、16res train 789 Do n't judge this place prima facie , you have to try it to believe it , a home away from home for the literate heart .####[['place', 'restaurant general', 'positive', 'try it and believe it']]
  15res dev 187、16res train 329 Service was devine , oysters where a sensual as they come , and the price ca n't be beat ! ! !####[['Service', 'service general', 'positive', 'devine'], ['oysters', 'food quality', 'positive', 'sensual'], ['NULL', 'restaurant prices', 'positive', "can't be beat"]]
  16res train 1008 DO not try unless you 're just going there to hang out like the rest of the hipsters who apparently have no sense of taste .####[['NULL', 'restaurant miscellaneous', 'negative', 'Do not try']]
  16res test 252 It was clear he didn ’ t really care .####[['NULL', 'service general', 'negative', 'didn \\’ t really care']]
  
  ```


# 数据分析


# 英文版MRC(dataset为ACOS和QUAD)
## 英文问题模板
```bash
Forward:

Q1:[CLS] What aspects ? [SEP] null sentence

Q2:[CLS] What opinions for the aspect ? [SEP] null sentence

Backward:

Q1:[CLS] What opinions ? [SEP] null sentence

Q2:[CLS] What aspects for the opinion ? [SEP] null sentence

Q3:[CLS] What category for the aspect and the opinion ? [SEP] sentence

Q4:[CLS] What sentiment for the aspect and the opinion ? [SEP] sentence
```


## Result
**ACOS-Restaurant**
|  | Precision | Recall | F1 |
| :----:| :----: | :----: | :----: |
| ACOS-Baseline | 38.54 | 52.96 | 44.61 |
| Span-ACOS | **64.37** | 48.91 | 55.58 |
| BMRC-ACOS(Ours) | 58.91 | 56.66 | 57.76 |
| Seq2Path(k=4) | - | - | 57.72 |
| Seq2Path(k=6) | - | - | 58.06 |
| Seq2Path(k=8) | - | - | 57.37 |
| Seq2Path(k=10) | - | - | **58.41** |
| BMRC-ACOS(Ours SentiWSP) | 60.90 | 60.70 | 60.80 |
| Tree Generation | **63.96** | **61.74** | **62.83** |

**QUAD**
| model | Res15-Precision | Res15-Recall | Res15-F1 | Res16-Precision | Res16-Recall | Res16-F1 |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: |
| HGCN-BERT + BERT-Linear | 0.2443 | 0.2025 | 0.2215 | 0.2536 | 0.2403 | 0.2468 |
| HGCN-BERT + BERT-TFM | 0.2555 | 0.2201 | 0.2365 | 0.2740 | 0.2641 | 0.2690 |
| TASO-BERT-Linear | 0.4186 | 0.2650 | 0.3246 | 0.4973 | 0.4070 | 0.4477 |
| TASO-BERT-CRF | 0.4424 | 0.2866 | 0.3478 | 0.4865 | 0.3968 | 0.4371 |
| GAS | 0.4531 | 0.4670 | 0.4598 | 0.5454 | 0.5762 | 0.5604 |
| PARAPHRASE | 0.4616 | 0.4772 | 0.4693 | 0.5663 | 0.5930 | 0.5793 |
| BMRC-ACOS(Ours) | **0.5353** | 0.4478 | 0.4877 | 0.5891 | 0.5708 | 0.5798 |
| BMRC-ACOS(Ours SentiWSP) | 0.4936 | **0.5371** | **0.5145** | **0.5910** | **0.6095** | **0.6001** |


# 中文版MRC(dataset为ASQE)
## 中文问题模板
```bash
Forward:

Q1:[CLS] 哪些方面？ [SEP] null sentence

Q2:[CLS] 哪些意见对于方面1？ [SEP] null sentence

Backward:

Q1:[CLS] 哪些意见？ [SEP] null sentence

Q2:[CLS] 哪些方面对于意见1？ [SEP] null sentence

Q3:[CLS] 什么情感对于方面1是意见2？ [SEP] sentence

Q4:[CLS] 什么种类对于方面1是意见2？ [SEP] sentence
```

我们划分数据集的方法是：总数据量：3199，train:dev:test = 8:1:1，按照顺序取前80%的数据作为训练集，10%作为验证集，10%作为测试集。
<font color="red">注：我们采用的计算方式与英文的一致，也就是严格计算方式！</font>

结果选取方式：所有epoch中best validation所对应的test result

<table>
  <tr>
    <td></td>
    <td colspan="3" align="center">Pair</td>
    <td colspan="3" align="center">Triplet</td>
    <td colspan="3" align="center">Quadruple</td>
  </tr>
  <tr>
    <td></td>
    <td align="center">Precision</td>
    <td align="center">Recall</td>
    <td align="center">F1</td>
    <td align="center">Precision</td>
    <td align="center">Recall</td>
    <td align="center">F1</td>
    <td align="center">Precision</td>
    <td align="center">Recall</td>
    <td align="center">F1</td>
  </tr>
  <tr>
    <td align="center"><B>Roberta+Bilstm</B></td>
    <td align="center">0.7831</td>
    <td align="center">0.7266</td>
    <td align="center">0.7538</td>
    <td align="center">0.7616</td>
    <td align="center">0.7055</td>
    <td align="center">0.7325</td>
    <td align="center">0.7483</td>
    <td align="center">0.6943</td>
    <td align="center">0.7203</td>
  </tr>
  <tr>
    <td align="center"><B>BMRC-ACOS(Ours chinese-bert-wwm)</B></td>
    <td align="center">0.7965</td>
    <td align="center">0.8295</td>
    <td align="center">0.8126</td>
    <td align="center">0.7832</td>
    <td align="center">0.8157</td>
    <td align="center">0.7991</td>
    <td align="center">0.7493</td>
    <td align="center">0.7803</td>
    <td align="center">0.7649</td>
  </tr>
    <tr>
    <td align="center"><B>BMRC-ACOS(Ours chinese-roberta-wwm-ext)</B></td>
    <td align="center"><B>0.8186</B></td>
    <td align="center"><B>0.8525</B></td>
    <td align="center"><B>0.8352</B></td>
    <td align="center"><B>0.8038</B></td>
    <td align="center"><B>0.8371</B></td>
    <td align="center"><B>0.8202</B></td>
    <td align="center"><B>0.7729</B></td>
    <td align="center"><B>0.8049</B></td>
    <td align="center"><B>0.7886</B></td>
  </tr>
</table>



# 论文数据分析表格

<table>
  <tr>
    <td></td>
    <td colspan="3" align="center">ASQP Rest15</td>
    <td colspan="3" align="center">ASQP Rest16</td>
    <td colspan="3" align="center">Chinese-ASQE</td>
  </tr>
  <tr>
    <td></td>
    <td align="center">train</td>
    <td align="center">dev</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">dev</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">dev</td>
    <td align="center">test</td>
  </tr>
  <tr>
    <td align="center"><B>句子数量</B></td>
    <td align="center">834</td>
    <td align="center">209</td>
    <td align="center">537</td>
    <td align="center">1264</td>
    <td align="center">316</td>
    <td align="center">544</td>
    <td align="center">2560</td>
    <td align="center">320</td>
    <td align="center">319</td>
  </tr>
  <tr>
    <td align="center"><B>句子最大长度</B></td>
    <td align="center">52</td>
    <td align="center">70</td>
    <td align="center">64</td>
    <td align="center">70</td>
    <td align="center">52</td>
    <td align="center">78</td>
    <td align="center">69</td>
    <td align="center">50</td>
    <td align="center">57</td>
  </tr>
  <tr>
    <td align="center"><B>显式四元组数量</B></td>
    <td align="center">1082</td>
    <td align="center">287</td>
    <td align="center">577</td>
    <td align="center">1543</td>
    <td align="center">403</td>
    <td align="center">620</td>
    <td align="center">1350</td>
    <td align="center">166</td>
    <td align="center">192</td>
  </tr>
  <tr>
    <td align="center"><B>隐式aspect四元组数量</B></td>
    <td align="center">272</td>
    <td align="center">60</td>
    <td align="center">218</td>
    <td align="center">1543</td>
    <td align="center">104</td>
    <td align="center">179</td>
    <td align="center">3766</td>
    <td align="center">481</td>
    <td align="center">444</td>
  </tr>
  <tr>
    <td align="center"><B>隐式opinion四元组数量</B></td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">146</td>
    <td align="center">10</td>
    <td align="center">15</td>
  </tr>
    <tr>
    <td align="center"><B>隐式aspect-opinion四元组数量</B></td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
    <td align="center">0</td>
  </tr>
  
</table>

