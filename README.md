# Deploying Atmospheric and Oceanic AI Models on Chinese Hardware and Framework: Migration Strategies, Performance Optimization and Analysis

This repository contains the code used for "Deploying Atmospheric and Oceanic AI Models on Chinese Hardware and Framework: Migration Strategies, Performance Optimization and Analysis" [[https://arxiv.org/pdf/2504.15322]](https://arxiv.org/abs/2510.17852)
The paper has been accepted by AAAI 2026 - IAAI.

## 📖 Introduction
With the growing role of artificial intelligence in climate and weather research, efficient model training and inference are in high demand. Current models like FourCastNet and AI-GOMS depend heavily on GPUs, limiting hardware independence, especially for Chinese domestic hardware and frameworks. To address this issue, we present a framework for migrating large-scale atmospheric and oceanic models from PyTorch to MindSpore and optimizing for Chinese chips, and evaluating their performance against GPUs. The framework focuses on software-hardware adaptation, memory optimization, and parallelism. Furthermore, the model's performance is evaluated across multiple metrics, including training speed, inference speed, model accuracy, and energy efficiency, with comparisons against GPU-based implementations. Experimental results demonstrate that the migration and optimization process preserves the models' original accuracy while significantly reducing system dependencies and improving operational efficiency by leveraging Chinese chips as a viable alternative for scientific computing. This work provides valuable insights and practical guidance for leveraging Chinese domestic chips and frameworks in atmospheric and oceanic AI model development, offering a pathway toward greater technological independence.

<img width="1471" height="691" alt="image" src="https://github.com/user-attachments/assets/92864632-e5f5-427a-b6ed-c1559aa9ec29" />


### Prerequisites
- Python 3.8+
- PyTorch 1.12+


## 🏃‍♂️ Training and Evaluation
1. Training
```bash
python train.py
```
2. Evaluation
```bash
python test.py
```
## 📜 Citation
waiting...

## 📧 Contact
For any questions or suggestions, please contact [Yuze Sun] at [syz23@mails.tsinghua.edu.cn] or open an issue on GitHub.

## 📄 License
This project is licensed under the MIT License.
