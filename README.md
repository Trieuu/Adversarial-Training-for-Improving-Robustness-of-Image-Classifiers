# Adversarial Training for Improving Robustness of Image Classifiers

This project investigates the effectiveness of **adversarial training** in improving the robustness of image classifiers against adversarial attacks.  
We use **MobileNetV3-Small** trained on the **CIFAR-100** dataset and evaluate how adversarial training with FGSM perturbations affects clean accuracy and adversarial robustness.  

The repository provides code for:
- **Baseline training** (clean images only)  
- **Adversarial training** (mix of clean and FGSM adversarial samples)  
- **Evaluation** on clean and adversarial test sets (Top-1 and Top-5 accuracy)  
- **Visualization** of dataset samples, preprocessing pipeline, and adversarial examples
- **In detail report** about the training and experiment process

---

## 📂 Repository Structure
<img width="1480" height="1398" alt="image" src="https://github.com/user-attachments/assets/c60f21eb-bf26-4f81-b137-3436ce4852b0" />



## ⚙️ Environment Setup

```bash
# 1. Create conda environment
conda create -n cifar100 python=3.10 -y
conda activate cifar100

# 2. Install dependencies
pip install -r requirements.txt
```

## 🚀 Training
You can train both the baseline and adversarial models using either the Colab notebooks or the scripts.

**Baseline (clean training)**
```bash
python -m scripts.train_baseline --epochs 10 --lr 5e-4 --eps 0.0078431373
```

**Adversarial Training**
```bash
python -m scripts.train_adv --epochs_adv 10 --lr 1e-3 --eps 0.0078431373 --adv_ratio 0.5 --init imagenet
```

**Google Colab**

- Baseline training notebook: ```mnv3_cifar100_baseline_train.ipynb```

- Adversarial training notebook: ```mnv3_cifar100_adversarial_train.ipynb```

Both notebooks are pre-configured to run on GPU (T4 in Colab).

## 📊 Visualization

Visualization utilities are available in ```visualize/report_visual.ipynb```, including:

- CIFAR-100 dataset samples

- Preprocessing & augmentation pipeline (resize, RandAugment, RandomErasing)

- Adversarial examples (clean vs adv vs perturbation)

- ε-sweep montage

- Per-class accuracy plots (clean vs adversarial)

Figures are automatically saved under ```run_mnv3/visuals/```.
