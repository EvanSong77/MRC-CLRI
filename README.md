# BMRC-ACOS
Use BMRC to complete ACOS tasks

# Contributions
- 我们在显式三元组ASTE任务中修改了原模型使其能做隐式四元组ACOS任务 最终结果比之前的bert baseline都高，比Extract-Classify-ACOS高**10.97**个点，精度也达到了最高的**64.37**。
- 我们首次用阅读理解模型来完成含隐式的四元组抽取任务。
- 我们模型BMRC-ACOS的效果比之前的baseline还有我们修改的Span-ACOS以及生成式Seq2Path(k=8)都高，结果为**57.76**，分别比Extract-Classify-ACOS、Span-ACOS、Seq2Path(k=8)高了**13.15**、**2.18**、**0.39**。

# Result
**Restaurant-ACOS**
|  | Precision | Recall | F1 |
| :----:| :----: | :----: | :----: |
| Double-Propagation-ACOS | 34.67 | 15.08 | 21.04 |
| JET-ACOS | 59.81 | 28.94 | 39.01 |
| Extract-Classify-ACOS | 38.54 | 52.96 | 44.61 |
| Span-ACOS | **64.37** | 48.91 | 55.58 |
| Seq2Path(k=8) | - | - | 57.37 |
| BMRC-ACOS(Ours) | 58.91 | **56.66** | **57.76** |
