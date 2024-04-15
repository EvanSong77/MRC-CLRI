
<h1 align="center">
Query-induced multi-task decomposition and enhanced learning for aspect-based sentiment quadruple prediction (MRC-CLRI)
</h1>

<div align="center">

![](https://img.shields.io/badge/Task-ASQP-orange)
![](https://img.shields.io/badge/Model-Released-blue)

</div>


<p align="center">
  <a href="#-introduction">Introduction</a> •
  <a href="#-data">Data</a> •
  <a href="#-quick-start">Quick Start</a>
</p>


## ✨ Introduction 
This repository contains the code and data for the paper titled "Query-induced multi-task decomposition and enhanced learning for aspect-based sentiment quadruple prediction". The paper introduces a novel end-to-end non-generative model for ASQP involving multi-task decomposition within machine reading comprehension (MRC) framework. This README provides an overview of the repository and instructions for running the code and using the data.


## 📃 Data
The ACOS dataset is sourced from [ACOS](https://github.com/NUSTM/ACOS/tree/main/data), while the ASQP dataset is sourced from [ABSA-QUAD](https://github.com/IsakZhang/ABSA-QUAD/tree/master/data).


## 🚀 Quick Start

### ⚙️ Setup
To run the code in this repository, you'll need the following dependencies:

- Python 3.9
- PyTorch 2.2
- transformers

Install these dependencies using pip:
```sh
conda create -n MRC-CLRI python=3.9
conda activate MRC-CLRI
pip install -r requirements.txt
```

### 🤖 Download Pre-trained Model
Before executing the code, you need to download the pre-trained model [SentiWSP](https://huggingface.co/shuaifan/SentiWSP/tree/main).

### ⚡️ Running the Code


- Model Training:

```sh
python run.py \
  --train_batch_size 4 \
  --data_path ./data/ASQP/rest15/v2/ \
  --task ASQP \
  --data_type rest15 \
  --model_path ../pretrained-models/SentiWSP \
  --do_train \
  --learning_rate1 3e-5 \
  --learning_rate2 1e-5 \
  --use_category_SCL \
  --use_sentiment_SCL
```


- Model Testing: 

We release the ASQP-Rest15 MRC-CLRI model (one seed): `rest15_test_model.pkl` [[Google Drive]](https://drive.google.com/file/d/14FKfm0TqGbcCwt6gUHgt2LqPLaFkARkF/view?usp=drive_link). You can run it with the following command:

```sh
# without Refined Inference
python run.py \
  --eval_batch_size 8 \
  --data_path ./data/ASQP/rest15/v2/ \
  --task ASQP \
  --data_type rest15 \
  --model_path ../pretrained-models/SentiWSP \
  --checkpoint_path ./outputs/saves/ASQP/rest15/rest15_test_model.pth \
  --do_test

# with Refined Inference
python run.py \
  --eval_batch_size 8 \
  --data_path ./data/ASQP/rest15/v2/ \
  --task ASQP \
  --data_type rest15 \
  --model_path ../pretrained-models/SentiWSP \
  --checkpoint_path ./outputs/saves/ASQP/rest15/rest15_test_model.pth \
  --do_test \
  --beta 25 \
  --alpha 0.82
```

- Model Inference:
```sh
python run.py \
  --do_inference \
  --load_ckpt_name ./outputs/saves/ASQP/rest15/rest15_test_model.pth
```