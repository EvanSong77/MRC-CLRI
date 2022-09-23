# BMRC-ACOS
Use BMRC to complete ACOS tasks

# Contributions
- 我们在显式三元组ASTE任务中修改了原模型使其能做隐式四元组ACOS任务 最终结果比之前的bert baseline都高，比Extract-Classify-ACOS高**10.97**个点，精度也达到了最高的**64.37**。
- 我们首次用阅读理解模型来完成含隐式的四元组抽取任务。
- 我们模型BMRC-ACOS的效果比之前的baseline还有我们修改的Span-ACOS以及生成式Seq2Path(k=8)都高，结果为**57.76**，分别比Extract-Classify-ACOS、Span-ACOS、Seq2Path(k=8)高了**13.15**、**2.18**、**0.39**。

# 英文版MRC
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
**Restaurant-ACOS**
|  | Precision | Recall | F1 |
| :----:| :----: | :----: | :----: |
| Double-Propagation-ACOS | 34.67 | 15.08 | 21.04 |
| JET-ACOS | 59.81 | 28.94 | 39.01 |
| Extract-Classify-ACOS | 38.54 | 52.96 | 44.61 |
| Span-ACOS | **64.37** | 48.91 | 55.58 |
| Seq2Path(k=8) | - | - | 57.37 |
| BMRC-ACOS(Ours) | 58.91 | **56.66** | **57.76** |
| Seq2Path(k=10) | - | - | **58.41** |


# 中文版MRC
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
    <td>0.8240</td>
    <td>0.7441</td>
    <td>0.7820</td>
    <td>0.7958</td>
    <td>0.7187</td>
    <td>0.7553</td>
  </tr>
  <tr>
    <td><B>BMRC-ACOS(Ours)</B></td>
    <td>0.8058</td>
    <td><B>0.8449</B></td>
    <td><B>0.8249</B></td>
    <td>0.7729</td>
    <td><B>0.8103</B></td>
    <td><B>0.7911</B></td>
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
    <td>0.7958</td>
    <td>0.7187</td>
    <td>0.7553</td>
  </tr>
  <tr>
    <td><B>BMRC-ACOS(Ours)</B></td>
    <td>0.7729</td>
    <td><B>0.8103</B></td>
    <td><B>0.7911</B></td>
  </tr>
</table>
