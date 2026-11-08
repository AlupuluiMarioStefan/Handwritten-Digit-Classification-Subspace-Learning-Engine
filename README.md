# Handwritten Digit Classification & Subspace Learning Engine

A Comparative Analysis of Linear Algebra Techniques for Handwritten Digit Recognition (USPS Dataset).

---

## 📌 Project Overview
This project investigates three different linear algebra methods for identifying handwritten digits ($0$ to $9$) from grayscale $16 \times 16$ pixel images using the US Postal Service dataset.

The primary goal is to evaluate the balance between computational efficiency and accuracy when scaling from single-vector representations to sub-space classifications.

---

## 📊 Dataset & Setup
- **Training Set:** 1,707 vectors in $\mathbb{R}^{256}$
- **Test Set:** 2,007 vectors in $\mathbb{R}^{256}$
- **Format:** Grayscale pixels normalized in $[-1, 1]$ ($1$ = black, $-1$ = white)
- **File:** `digitimagesandlabels.mat`, containing `trainingimages`, `trainingimageslabels`, `testimages`, and `testimageslabels`

---

## 🛠️ Algorithms & Methodology

1. **Nearest Average (Baseline):**
   - Classifies images by calculating the Euclidean distance to the mean vector of each digit class.
   - *Pros/Cons:* Extremely fast, but blurs distinctive handwriting styles into a single mean image.

2. **Nearest Neighbor:**
   - Compares test images against all 1,707 training images.
   - *Pros/Cons:* Captures style variations, but requires high memory and full search runtime ($O(N)$).

3. **SVD Subspace Classifier (Truncated SVD):**
   - Uses Economy SVD ($A_j = U_j \Sigma_j V_j^T$) to build a $r=15$ dimensional basis subspace ($Q_j$) for each digit class.
   - Classifies images by minimizing residual projection errors: $\min_j \|z - Q_j Q_j^T z\|_2$.
   - *Pros/Cons:* Best balance of high accuracy and low execution time.

---

## 📈 Results & Key Findings

| Method | Correct / Total | Accuracy (%) | Execution Time (s) |
| :--- | :---: | :---: | :---: |
| **Nearest Average** | 1623 / 2007 | **80.87%** | **0.02 s** |
| **Nearest Neighbor** | 1838 / 2007 | **91.58%** | 1.00 s |
| **SVD Subspace ($r=15$)** | **1889 / 2007** | **94.12%** | **0.08 s** |

### Key Takeaways:
- **SVD Subspace** achieved the highest accuracy (**94.12%**) while being **12.5x faster** than Nearest Neighbor.
- Most misclassifications occur on geometrically similar digits (e.g., $4 \rightarrow 9$ and $0 \rightarrow 6$).

![Confusion matrices for all three methods](confusion_matrices.png)

---

## 📂 Project Structure
This repository is currently a flat set of files:
- `recognizedigits.m` — loads the data and runs all three classifiers, printing accuracy/timing results and plotting confusion matrices.
- `plotdigit.m` — helper function that renders a single $256 \times 1$ image vector as a $16 \times 16$ digit.
- `digitimagesandlabels.mat` — the training/test data described above.
- `confusion_matrices.png` — confusion matrix figure for all three methods (Figure 1 in the report).
- `Handwritten_Digit_Classification___Subspace_Learning_Engine.pdf` — the full project report.

If you'd like to mirror the layout described in earlier drafts of this README, you can instead organize it as:
```
src/     recognizedigits.m, plotdigit.m
docs/    the PDF report
assets/  confusion_matrices.png
```
and update the `load`/`addpath` calls in `recognizedigits.m` accordingly.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/nla-digit-classification.git
   cd nla-digit-classification
   ```
2. Open MATLAB (or GNU Octave) and set the repository folder as your working directory, or add it to your path:
   ```matlab
   addpath('path/to/nla-digit-classification')
   ```
3. Run the main script:
   ```matlab
   recognizedigits
   ```
   This loads `digitimagesandlabels.mat`, trains and evaluates all three classifiers, prints accuracy/timing/per-digit results to the console, and plots the confusion matrices shown above.
4. (Optional) Visualize an individual digit vector, e.g. the first training image:
   ```matlab
   load digitimagesandlabels
   plotdigit(trainingimages(:, 1))
   ```

---

## ✅ Reproducibility
The results above were independently cross-checked against a reference NumPy implementation using the same `.mat` file; the overall accuracy, per-digit accuracy, and misclassification counts all matched exactly.
