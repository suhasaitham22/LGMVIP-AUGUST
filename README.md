# LGMVIP August (LetsGrowMore Virtual Internship)

LetsGrowMore virtual internship project work from August 2022. Two classic machine learning notebooks: flower classification on the Iris dataset and handwritten digit classification on MNIST.

## Notebooks

### Iris Flowers Classification

`LGMVIP IRIS CLASSIFICATION.ipynb` with `Iris.csv` (150 rows: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species).

- **EDA:** dataset info, descriptive statistics, correlation matrix, outlier checks, countplot of the three species, and a pairplot of the features.
- **Modeling:** label-encode the species, split 80/20 (random_state=1), and train a `DecisionTreeClassifier(max_depth=8)`.
- **Visualization:** PCA down to 2 components with a plotted decision boundary showing how the tree separates the three species.
- **Result:** accuracy 0.9667 on the test set.

### MNIST Handwritten Digit Classification

`MNIST Handwritten Digit Classification.ipynb` with `train.csv` (42,000 rows: a label column plus 784 pixel columns for 28x28 grayscale digit images).

- **Data prep:** separate the label from the pixel features, split 37,800 train / 4,200 test.
- **Modeling:** compare SVM kernels, then grid-search C and gamma.
- **Results:** linear SVM 0.9043, RBF SVM 0.9251, and the final tuned model `SVC(C=1, gamma=0.001, kernel='rbf')` scoring 0.925 on the test set (about 92.5%).

## How to run

```bash
git clone https://github.com/suhasaitham22/LGMVIP-AUGUST.git
cd LGMVIP-AUGUST
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Open either notebook and run the cells top to bottom. The CSVs are already in the repo.

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter Notebook.
