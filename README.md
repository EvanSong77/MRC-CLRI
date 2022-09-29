# BMRC-ACOS
Use BMRC to complete Aspect Sentiment Quadruple Extraction tasks in ACOS(rest,laptop), QUAD(rest15,rest16) and ASQE dataset.

# Contributions
- 我们的模型巧妙的解决了aspect或者opinion中含有隐式的情况，将含隐式抽取问题转化为“显式”抽取
- 我们的MRC模型既可以适用英文也可以适用中文，针对中英文设计出了中英文两种模板，且在中文效果显著，英文的效果不逊色T5生成式模型
- 我们的模型在QUAD的两个数据集都达到了SOTA，在ASQE数据集也达到SOTA（超过此前SOTA 严格(4.01%) 不严格(6.69%)）

# 数据分析


# 英文版MRC(dataset为ACOS和QUAD)
## 英文问题模板
```bash
Forward:

Q1:[CLS] What aspects ? [SEP] null sentence

Q2:[CLS] What opinions given the aspect1 ? [SEP] null sentence

Backward:

Q1:[CLS] What opinions ? [SEP] null sentence

Q2:[CLS] What aspects given the opinion1 ? [SEP] null sentence

Q3:[CLS] What category given the aspect1 and the opinion1 ? [SEP] sentence

Q4:[CLS] What sentiment given the aspect1 and the opinion1 ? [SEP] sentence
```

## Result
**ACOS-Restaurant**
|  | Precision | Recall | F1 |
| :----:| :----: | :----: | :----: |
| Double-Propagation-ACOS | 34.67 | 15.08 | 21.04 |
| JET-ACOS | 59.81 | 28.94 | 39.01 |
| Extract-Classify-ACOS | 38.54 | 52.96 | 44.61 |
| Span-ACOS | **64.37** | 48.91 | 55.58 |
| Seq2Path(k=8) | - | - | 57.37 |
| BMRC-ACOS(Ours) | 58.91 | **56.66** | **57.76** |
| Seq2Path(k=10) | - | - | **58.41** |
| Tree Generation | **63.96** | **61.74** | **62.83** |

**QUAD**
| model | Res15-Precision | Res15-Recall | Res15-F1 | Res16-Precision | Res16-Recall | Res16-F1 |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: |
| HGCN-BERT + BERT-Linear | 0.2443 | 0.2025 | 0.2215 | 0.2536 | 0.2403 | 0.2468 |
| HGCN-BERT + BERT-TFM | 0.2555 | 0.2201 | 0.2365 | 0.2740 | 0.2641 | 0.2690 |
| TASO-BERT-Linear | 0.4186 | 0.2650 | 0.3246 | 0.4973 | 0.4070 | 0.4477 |
| TASO-BERT-CRF | 0.4424 | 0.2866 | 0.3478 | 0.4865 | 0.3968 | 0.4371 |
| GAS | 0.4531 | 0.4670 | 0.4598 | 0.5454 | 0.5762 | 0.5604 |
| PARAPHRASE | 0.4616 | **0.4772** | 0.4693 | 0.5663 | **0.5930** | **0.5793** |
| BMRC-ACOS(Ours) | **0.5353** | 0.4478 | **0.4877** | **0.5891** | 0.5670 | 0.5778 |
| BMRC-ACOS(Ours不严格计算) | 0.5698 | 0.4679 | 0.5138 | 0.6216 | 0.5982 | 0.6097 |
BMRC-ACOS从57.5->57.78

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

## 严格计算结果
<table>
  <tr>
    <td></td>
    <td colspan="3" align="center">Pair</td>
    <td colspan="3" align="center">Quadruple</td>
  </tr>
  <tr>
    <td></td>
    <td align="center">pre</td>
    <td align="center">recall</td>
    <td align="center">f1</td>
    <td align="center">pre</td>
    <td align="center">recall</td>
    <td align="center">f1</td>
  </tr>
  <tr>
    <td><B>Roberta+Bilstm</B></td>
    <td><B>0.8240</B></td>
    <td>0.7441</td>
    <td>0.7820</td>
    <td><B>0.7958</B></td>
    <td>0.7187</td>
    <td>0.7553</td>
  </tr>
  <tr>
    <td><B>BMRC-ACOS(Ours chinese-bert-wwm)</B></td>
    <td>0.8058</td>
    <td>0.8449</td>
    <td><B>0.8249</B></td>
    <td>0.7729</td>
    <td>0.8103</td>
    <td>0.7911</td>
  </tr>
  <tr>
    <td><B>BMRC-ACOS(Ours chinese-roberta-wwm-ext)</B></td>
    <td>0.8038</td>
    <td><B>0.8458</B></td>
    <td>0.8243</td>
    <td>0.7756</td>
    <td><B>0.8162</B></td>
    <td><B>0.7954</B></td>
  </tr>
</table>

## 非严格计算结果（师兄计算方式）
<table>
  <tr>
    <td></td>
    <td colspan="3" align="center">Quadruple</td>
  </tr>
  <tr>
    <td></td>
    <td align="center">pre</td>
    <td align="center">recall</td>
    <td align="center">f1</td>
  </tr>
  <tr>
    <td><B>Roberta+Bilstm</B></td>
    <td>0.8619</td>
    <td>0.7865</td>
    <td>0.8224</td>
  </tr>
  <tr>
    <td><B>BMRC-ACOS(Ours chinese-bert-wwm)</B></td>
    <td>0.8582</td>
    <td><B>0.9150</B></td>
    <td>0.8857</td>
  </tr>
    <tr>
    <td><B>BMRC-ACOS(Ours chinese-roberta-wwm-ext)</B></td>
    <td><B>0.8668</B></td>
    <td>0.9130</td>
    <td><B>0.8893</B></td>
  </tr>
</table>


# 中文版划分数据集之后
我们划分数据集的方法是：总数据量：3199，train:dev:test = 8:1:1，随机打乱后取其中的80%作为训练集，10%作为验证集，10%作为测试集。
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
    <td align="center"><B>0.8295</B></td>
    <td align="center">0.8126</td>
    <td align="center">0.7832</td>
    <td align="center"><B>0.8157</B></td>
    <td align="center">0.7991</td>
    <td align="center">0.7493</td>
    <td align="center"><B>0.7803</B></td>
    <td align="center">0.7649</td>
  </tr>
    <tr>
    <td align="center"><B>BMRC-ACOS(Ours chinese-roberta-wwm-ext)</B></td>
    <td align="center"><B>0.8305</B></td>
    <td align="center">0.8280</td>
    <td align="center"><B>0.8292</B></td>
    <td align="center"><B>0.8166</B></td>
    <td align="center">0.8141</td>
    <td align="center"><B>0.8154</B></td>
    <td align="center"><B>0.7797</B></td>
    <td align="center">0.7773</td>
    <td align="center"><B>0.7785</B></td>
  </tr>
</table>

