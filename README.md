# Financial-ICS

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/Jintao-Huang/financial-ics/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Jintao-Huang/financial-ics"></a>
<a href="https://github.com/Jintao-Huang/financial-ics/"><img src="https://img.shields.io/badge/financial--ics-Build from source-6FEBB9.svg"></a>
</p>

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Installation](#-installation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Demo](#-demo)
- [License](#license)


## 📝 Introduction
Financial-ICS is an algorithm-based industry classification system (ICS) that utilizes 10K reports as textual corpora to identify peer firms that are economically related to the focal firm. ICSs play a crucial role in financial analysis, such as accurately assessing market value, estimating competition and risks, determining executive compensation, and gaining a deeper understanding of strategic behavior.

We utilize the "Item 1 Business" and "Item 1A Risk Factors" sections from 10K reports as inputs to our model. We use attention-optimized LongBert to generate product representations corresponding to each company. Peer firms are identified based on the cosine similarity between the product representations of different companies.

We have designed the **LongBert** model, which can directly process text with up to 131K tokens, meaning it can handle any whole document in the dataset. LongBert is based on RoBERTa and modifies the positional encoding and self-attention methods. Therefore, it can reuse almost all parameters from the pre-trained RoBERTa checkpoint. LongBert utilizes rotary position embedding as the positional encoding method. Furthermore, it combines shifted block attention and global attention, leveraging the principle of information locality in natural language while supplementing the model's ability to learn long-range dependencies with global attention. This reduces the attention complexity from $O(n^2)$ to $O(n)$.

Our training framework employs a fully unsupervised approach. Firstly, we use a masked language model loss for continued pre-training. Then, we use contrastive learning methods to address the anisotropy problem present in pre-trained language models.

We have made the training dataset, code, and model weights publicly available.


## 🛠️ Installation
```bash
git clone https://github.com/Jintao-Huang/financial-ics.git
cd financial-ics
pip install -e .
```

Downloading the Dataset

```bash
git clone https://www.modelscope.cn/datasets/swift/finance_tenk.git
cd finance_tenk
tar -xf data.tar
cd ..
```


## 🚀 Training
MaskedLM Training

```bash
bash scripts/lbert/maskedlm_train.sh
# ddp
bash scripts/lbert/maskedlm_train_ddp.sh
```

Contrastive Learning

```bash
bash scripts/lbert/cl_train.sh
# ddp
bash scripts/lbert/cl_train_ddp.sh
```

## 🎯 Evaluation

```bash
bash scripts/lbert/eval.sh
```

## ✨ Demo

```bash
bash scripts/lbert/app.sh
```

## License

This project is licensed under the [MIT](https://github.com/Jintao-Huang/financial-ics/blob/master/LICENSE).

