# Horse vs Human Classifier  
## CNNs from Scratch vs Transfer Learning

Binary image classification project comparing convolutional neural networks trained from scratch against transfer learning approaches.

Developed by Lorenzo Ferrer de Oya and Ariel Núñez Valencia  
Artificial Vision — Universidad Alfonso X el Sabio (UAX)

---

## Overview

This project explores how different modeling strategies perform under limited data conditions (~800 training images).

We implemented:

- CNN architectures from scratch  
- Transfer Learning (VGG-16, EfficientNetB0)  
- Model evaluation (Accuracy, Recall, AUC, Brier Score)  
- Cost-sensitive threshold tuning  
- Grad-CAM for interpretability  

Dataset: horses_or_humans (TensorFlow Datasets)

---

## Problem Context

Not all errors have the same impact:

- Missing a horse → high cost (safety risk)  
- False positive → low cost  

This introduces a cost-sensitive classification problem, where recall is more important than precision.

---

## Key Insights

### 1. CNNs from scratch overfit easily
- Deep models collapse with small datasets  
- Train accuracy ~0.97 vs validation ~0.50  
- Simpler architectures generalize better  

---

### 2. Transfer Learning dominates
- VGG-16 → AUC = 1.00  
- EfficientNetB0 → AUC = 1.00  
- Pretrained ImageNet features drastically improve performance  

---

### 3. Threshold tuning matters
Adjusting the classification threshold:

- Recall increases from 0.578 to 0.664  
- F2-score improves  
- Operational cost decreases by ~18%  

Model decisions are as important as model training.

---

### 4. Interpretability is key
Using Grad-CAM:

- CNN-1 focuses on horse silhouette → meaningful learning  
- CNN-2 / CNN-3 show diffuse attention → poor feature extraction  

---

## Model Comparison

| Model            | Accuracy | Recall (Horse) | AUC  | Parameters |
|------------------|----------|----------------|------|------------|
| CNN-1 Baseline   | 0.789    | 0.578          | 0.969| 5.4M       |
| CNN-2 Deep + BN  | 0.500    | 0.000          | 0.156| 0.3M       |
| CNN-3 Depthwise  | 0.500    | 0.000          | 0.818| 0.1M       |
| VGG-16           | 1.000    | 1.000          | 1.000| 14.8M      |
| EfficientNetB0   | 1.000    | 1.000          | 1.000| 4.2M       |

---

## Techniques Used

- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Binary Cross-Entropy Loss
- Adam / SGD / RMSprop optimizers
- Threshold optimization (cost-sensitive)
- Grad-CAM (model interpretability)

---

## Repository Structure


notebook/
├── horse_vs_human.ipynb
└── horse_vs_human.html

images/
├── gradcam.png
└── metrics.png


---

## Full Notebook

The complete notebook (with code, outputs, and visualizations) is available here:

notebook/horse_vs_human.html

---

## Takeaways

- Small datasets require careful model selection  
- Transfer learning is often the practical standard  
- Threshold tuning can significantly impact real-world performance  
- Interpretability helps validate model behavior  

---

## Future Work

- Multi-class classification (horse / human / rider)  
- Video-based detection (temporal models)  
- Uncertainty estimation (MC Dropout)  
- Larger and more diverse datasets  

---

## Contact

If you are working on computer vision or machine learning projects, feel free to connect.
