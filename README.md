# Introduction to Artificial Intelligence

My coursework and tutorial work for **Introduction to Artificial Intelligence** module.

## Tutorials

| # | Topic | What I did |
|---|-------|------------|
| 1 | [Python and data](tutorial1/) | Python basics, pandas on the Auto MPG dataset, loop and iris exercises |
| 2 | [Classification with scikit-learn](tutorial2/) | Perceptron on breast cancer, decision trees and k-fold on iris and wine, ID3 from scratch on zoo |
| 3 | [Regression and SVMs](tutorial3/) | Linear regression on auto-mpg and diabetes (RMSE, R²), SVMs with kernels and soft margins, one-vs-one SVM on iris, SVR |
| 4 | [Unsupervised learning](tutorial4/) | PCA for visualisation, denoising and speeding up an SVM, k-means and Gaussian mixture clustering, PCA + k-means + GMM on iris, PCA + SVM face recognition (85% accuracy) |

## Tools

- Python (Anaconda distribution)
- Jupyter Notebook and Spyder
- pandas, NumPy, SciPy, scikit-learn, matplotlib
- TensorFlow (later in the module)

## Setup

The environment is in `intro_ai_env.yml` (Python 3.11):

    conda env create -f intro_ai_env.yml
    conda activate intro_ai
    conda install pandas

## Running the notebooks

Replace tutorial1 with tutorial2 (and so on) in the cd command below.

```bash
git clone https://github.com/danzgeorg/intro-to-ai.git
cd intro-to-ai/tutorial1
jupyter notebook
```

Each notebook expects its data file to be in the same folder.
