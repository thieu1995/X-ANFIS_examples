# X-ANFIS_examples
Code for case studies in paper "X-ANFIS: An Extensible and Cross-Learning ANFIS Framework for Machine Learning Tasks"


1) Compared models

```code
# Support Vector Machine
# K-Nearest Neighbors
# Decision Tree
# Random Forest
# Gradient Boosting
# Multi-Layer Perceptron
# Traditional Anfis
# Hybrid Adam-trained Anfis
# Hybrid RMSprop-trained Anfis
# Hybrid Adadelta-trained Anfis
# Bio SHADE-trained Anfis
# Bio L-SHADE-trained Anfis
# Bio IM-ARO-trained Anfis 
# Bio QLE-SCA-trained Anfis
# Bio A-EO-trained Anfis
# Bio AIW-PSO-trained Anfis



classifiers = {
    "SVC": SVC(kernel="linear", C=0.1, random_state=42),
    "DTC": DecisionTreeClassifier(max_depth=5, random_state=42),
    "RFC": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2, random_state=42),
    "GBC": GradientBoostingClassifier(n_estimators=75, learning_rate=0.5, max_depth=1, random_state=42),
    "MLP": MLPClassifier(alpha=1, max_iter=750, hidden_layer_sizes=(15,), activation="relu", random_state=42),
    "ANFIS": AnfisClassifier(expand_name="chebyshev", n_funcs=3, act_name="sigmoid",
                   obj_name="BCEL", max_epochs=750, batch_size=32, optimizer="SGD", verbose=False)
}
# ANFIS
# GA-ANFIS       Genetic Algorithm
# AVOA-ANFIS     Artificial Gorilla Troops Optimization
# ARO-ANFIS      Artificial Rabbits Optimization
# CDO-ANFIS      Chernobyl Disaster Optimization
# RUN-ANFIS      Success History Intelligent Optimization
# INFO-ANFIS     weIghted meaN oF vectOrs


```

2) Dataset

```code 
1. BreastEW: Binary classification problem: https://github.com/thieu1995/MetaCluster
2. Heart:  https://github.com/thieu1995/MetaCluster
3. Iris: Multiclass classification problem: https://github.com/thieu1995/MetaCluster
4. Wine: https://github.com/thieu1995/MetaCluster
5. Banknote: https://github.com/thieu1995/MetaCluster
6. MovieRevenue: https://www.kaggle.com/competitions/tmdb-box-office-prediction/data
    + Preprocessing: https://www.kaggle.com/code/karansehgal13/random-forest-regressor-for-tmdb-b-o-prediction/notebook
7. Concrete: UCI
8. Energy: UCI
9. Abalone: UCI
10. RealEstate: UCI
```
