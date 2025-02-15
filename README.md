# DataScience x Régression Logistique - Harry Potter & Data Scientist

Ce projet consiste à créer un modèle de régression logistique (one-vs-all) en Python pour attribuer à chaque élève de Poudlard sa maison (Gryffindor, Hufflepuff, Ravenclaw, Slytherin).

---

## Fichiers Principaux

- **describe.py**  
  Analyse un fichier CSV (dataset_train.csv) et affiche, sans fonctions prédéfinies, les statistiques (count, moyenne, écart-type, min, 25%, 50%, 75%, max) de chaque feature numérique.

- **histogram.py**  
  Affiche un histogramme pour identifier le cours dont la distribution des scores est homogène entre les maisons.

- **scatter_plot.py**  
  Génère un scatter plot afin de repérer les deux features les plus similaires.

- **pair_plot.py**  
  Crée une matrice de scatter plots pour visualiser les relations entre features et sélectionner celles à utiliser pour la régression logistique.

- **logreg_train.py**  
  Entraîne le modèle de régression logistique avec descente de gradient sur dataset_train.csv et sauvegarde les poids dans un fichier CSV.

- **logreg_predict.py**  
  Utilise les poids sauvegardés pour prédire les maisons sur dataset_test.csv et génère un fichier `houses.csv` au format suivant :

Index,Hogwarts House 0,Gryffindor 1,Hufflepuff 2,Ravenclaw ...


---

## Utilisation

1. **Analyse des données :**
 ```bash
 python describe.py dataset_train.csv
```
    Visualisations :
        Histogramme :

python histogram.py dataset_train.csv

Scatter plot :

python scatter_plot.py dataset_train.csv

Pair plot :

    python pair_plot.py dataset_train.csv

Entraînement du modèle :

python logreg_train.py dataset_train.csv

Prédiction :

    python logreg_predict.py dataset_test.csv weights.csv


