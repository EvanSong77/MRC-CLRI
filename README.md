# BMRC-ACOS
Use BMRC to complete Aspect Sentiment Quadruple Extraction tasks in ACOS(rest,laptop), QUAD(rest15,rest16) and ASQE dataset.

# Contributions

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

**QUAD**
| model | Res15-Precision | Res15-Recall | Res15-F1 | Res16-Precision | Res16-Recall | Res16-F1 |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: |
| HGCN-BERT + BERT-Linear | 0.2443 | 0.2025 | 0.2215 | 0.2536 | 0.2403 | 0.2468 |
| HGCN-BERT + BERT-TFM | 0.2555 | 0.2201 | 0.2365 | 0.2740 | 0.2641 | 0.2690 |
| TASO-BERT-Linear | 0.4186 | 0.2650 | 0.3246 | 0.4973 | 0.4070 | 0.4477 |
| TASO-BERT-CRF | 0.4424 | 0.2866 | 0.3478 | 0.4865 | 0.3968 | 0.4371 |
| GAS | 0.4531 | 0.4670 | 0.4598 | 0.5454 | 0.5762 | 0.5604 |
| PARAPHRASE | 0.4616 | **0.4772** | 0.4693 | 0.5663 | **0.5930** | **0.5793** |
| BMRC-ACOS(Ours) | **0.5353** | 0.4478 | **0.4877** | **0.5794** | 0.5707 | 0.5750 |


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
