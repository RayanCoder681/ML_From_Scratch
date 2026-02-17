"""
╔══════════════════════════════════════════════════════════════════╗
║            KNN (K-Nearest Neighbors) FROM SCRATCH               ║
║                  Implémentation propre & réutilisable            ║
╚══════════════════════════════════════════════════════════════════╝

Auteur : [Ton Nom]
Date   : Février 2026
But    : Implémenter KNN sans aucune librairie ML — uniquement NumPy

📌 L'idée de KNN est ultra simple :
   "Pour classer un nouveau point, regarde ses K voisins les plus proches
    et vote pour la classe majoritaire."
"""

import numpy as np
from collections import Counter


# ============================================================
# 🧮 FONCTIONS DE DISTANCE
# ============================================================

def distance_euclidienne(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calcule la distance euclidienne entre deux points.
    
    Formule mathématique :
        d(x1, x2) = √( Σ (x1_i - x2_i)² )
    
    C'est la distance "classique" en ligne droite entre deux points.
    
    Exemple :
        x1 = [1, 2], x2 = [4, 6]
        d = √((4-1)² + (6-2)²) = √(9 + 16) = √25 = 5.0
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def distance_manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calcule la distance de Manhattan (L1) entre deux points.
    
    Formule :
        d(x1, x2) = Σ |x1_i - x2_i|
    
    Imagine que tu marches dans une ville en grille (comme Manhattan) :
    tu ne peux pas couper en diagonale, tu dois tourner aux coins.
    
    Exemple :
        x1 = [1, 2], x2 = [4, 6]
        d = |4-1| + |6-2| = 3 + 4 = 7
    """
    return np.sum(np.abs(x1 - x2))


def distance_minkowski(x1: np.ndarray, x2: np.ndarray, p: int = 2) -> float:
    """
    Calcule la distance de Minkowski (généralisation).
    
    Formule :
        d(x1, x2) = (Σ |x1_i - x2_i|^p)^(1/p)
    
    - p=1 → Manhattan
    - p=2 → Euclidienne
    - p=∞ → Chebyshev (max des différences)
    """
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1 / p)


# ============================================================
# 🧠 CLASSE KNN — L'ALGORITHME COMPLET
# ============================================================

class KNN:
    """
    K-Nearest Neighbors Classifier — implémenté from scratch.
    
    Comment ça marche (en 3 étapes) :
    ─────────────────────────────────
    1. STOCKER : On mémorise toutes les données d'entraînement
       (KNN est un "lazy learner" — il n'apprend rien pendant fit!)
    
    2. CALCULER : Pour chaque nouveau point, on calcule sa distance
       avec TOUS les points d'entraînement
    
    3. VOTER : On prend les K points les plus proches et on vote
       pour la classe majoritaire
    
    Paramètres :
    ────────────
    k : int
        Nombre de voisins à considérer.
        - k petit (ex: 1-3) → sensible au bruit, frontières complexes
        - k grand (ex: 15-20) → plus lisse, mais peut ignorer les détails
        - Règle : k doit être IMPAIR pour éviter les égalités
    
    metric : str
        La mesure de distance à utiliser ('euclidean', 'manhattan', 'minkowski')
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        # Vérification : k doit être un entier positif
        if k < 1:
            raise ValueError(f"k doit être ≥ 1, reçu : {k}")
        
        self.k = k
        self.metric = metric
        
        # On choisit la fonction de distance
        self._distance_functions = {
            'euclidean': distance_euclidienne,
            'manhattan': distance_manhattan,
            'minkowski': distance_minkowski
        }
        
        if metric not in self._distance_functions:
            raise ValueError(f"Métrique inconnue : {metric}. "
                           f"Choix possibles : {list(self._distance_functions.keys())}")
        
        self._distance_fn = self._distance_functions[metric]
        
        # Ces attributs seront remplis par fit()
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        "Entraîner" le modèle = simplement stocker les données.
        
        C'est ce qui fait de KNN un "lazy learner" (apprenant paresseux).
        Pas de calcul pendant l'entraînement !
        
        Paramètres :
            X : np.ndarray de shape (n_samples, n_features)
                Les données d'entraînement (features)
            y : np.ndarray de shape (n_samples,)
                Les étiquettes (classes)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Vérification : k ne doit pas dépasser le nombre d'échantillons
        if self.k > len(self.X_train):
            raise ValueError(
                f"k={self.k} est plus grand que le nombre "
                f"d'échantillons d'entraînement ({len(self.X_train)})"
            )
        
        return self  # Pour permettre le chaînage : model.fit(X, y).predict(X_test)
    
    def _predict_single(self, x: np.ndarray) -> str:
        """
        Prédire la classe d'UN SEUL point.
        
        Étapes :
        1. Calculer la distance entre x et TOUS les points d'entraînement
        2. Trier par distance croissante
        3. Prendre les K plus proches
        4. Voter : la classe la plus fréquente gagne
        """
        # Étape 1 : Calculer toutes les distances
        distances = np.array([
            self._distance_fn(x, x_train) 
            for x_train in self.X_train
        ])
        
        # Étape 2 : Obtenir les indices des K plus petites distances
        # np.argsort() retourne les indices qui trieraient le tableau
        k_indices = np.argsort(distances)[:self.k]
        
        # Étape 3 : Récupérer les classes de ces K voisins
        k_voisins_labels = self.y_train[k_indices]
        
        # Étape 4 : Vote majoritaire
        # Counter({'Iris-setosa': 3, 'Iris-versicolor': 2}) → 'Iris-setosa'
        vote = Counter(k_voisins_labels)
        classe_predite = vote.most_common(1)[0][0]
        
        return classe_predite
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédire les classes pour un ensemble de points.
        
        Paramètres :
            X : np.ndarray de shape (n_samples, n_features)
        
        Retourne :
            np.ndarray de shape (n_samples,) avec les classes prédites
        """
        if self.X_train is None:
            raise RuntimeError("Le modèle n'a pas été entraîné ! Appelle fit() d'abord.")
        
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> dict:
        """
        Renvoie les probabilités de chaque classe pour chaque point.
        
        La "probabilité" = proportion des K voisins appartenant à chaque classe.
        Exemple avec k=5 : si 3 voisins sont Setosa et 2 sont Versicolor,
                           → P(Setosa) = 3/5 = 0.6, P(Versicolor) = 2/5 = 0.4
        """
        all_probas = []
        
        for x in X:
            distances = np.array([
                self._distance_fn(x, x_train)
                for x_train in self.X_train
            ])
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            
            vote = Counter(k_labels)
            probas = {
                label: count / self.k 
                for label, count in vote.items()
            }
            all_probas.append(probas)
        
        return all_probas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculer l'accuracy (précision globale) du modèle.
        
        Accuracy = nombre de prédictions correctes / nombre total de prédictions
        """
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy
    
    def __repr__(self):
        return f"KNN(k={self.k}, metric='{self.metric}')"


# ============================================================
# 📊 FONCTIONS D'ÉVALUATION — FROM SCRATCH
# ============================================================

def train_test_split_manual(X, y, test_size=0.2, random_seed=42):
    """
    Séparer les données en ensemble d'entraînement et de test.
    
    Pourquoi ? Pour évaluer le modèle sur des données qu'il n'a JAMAIS vues.
    Si on teste sur les mêmes données qu'on entraîne → TRICHE !
    
    Paramètres :
        X : features
        y : labels
        test_size : proportion pour le test (0.2 = 20%)
        random_seed : pour la reproductibilité
    """
    np.random.seed(random_seed)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Mélanger les indices
    indices = np.random.permutation(n_samples)
    
    # Séparer
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def matrice_de_confusion(y_true, y_pred):
    """
    Construire la matrice de confusion from scratch.
    
    La matrice de confusion montre :
    - Lignes = classes réelles
    - Colonnes = classes prédites
    - Diagonale = prédictions correctes ✅
    - Hors diagonale = erreurs ❌
    
    Retourne :
        matrix : np.ndarray (n_classes x n_classes)
        classes : list des noms de classes (dans l'ordre)
    """
    classes = sorted(list(set(y_true) | set(y_pred)))
    n_classes = len(classes)
    
    # Créer un mapping classe → index
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Remplir la matrice
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        i = class_to_idx[true]
        j = class_to_idx[pred]
        matrix[i][j] += 1
    
    return matrix, classes


def accuracy_score(y_true, y_pred):
    """Accuracy = prédictions correctes / total"""
    return np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)


def classification_report_manual(y_true, y_pred):
    """
    Générer un rapport de classification complet (comme sklearn).
    
    Pour chaque classe, calcule :
    - Précision : parmi les prédictions de cette classe, combien sont correctes ?
    - Rappel : parmi les vrais exemples de cette classe, combien ont été trouvés ?
    - F1-Score : moyenne harmonique de Précision et Rappel
    """
    classes = sorted(list(set(y_true) | set(y_pred)))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    report = {}
    for cls in classes:
        # True Positives : correctement prédit comme cls
        tp = np.sum((y_true == cls) & (y_pred == cls))
        # False Positives : prédit comme cls mais c'est faux
        fp = np.sum((y_true != cls) & (y_pred == cls))
        # False Negatives : est cls mais prédit autrement
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == cls)
        
        report[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(support)
        }
    
    return report


def print_classification_report(report):
    """Afficher le rapport de classification de manière formatée."""
    print(f"\n{'Classe':<25} {'Précision':>10} {'Rappel':>10} {'F1-Score':>10} {'Support':>10}")
    print("─" * 70)
    
    total_support = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    
    for cls, metrics in report.items():
        print(f"{cls:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10d}")
        total_support += metrics['support']
        avg_precision += metrics['precision'] * metrics['support']
        avg_recall += metrics['recall'] * metrics['support']
        avg_f1 += metrics['f1'] * metrics['support']
    
    print("─" * 70)
    print(f"{'Weighted Avg':<25} {avg_precision/total_support:>10.4f} "
          f"{avg_recall/total_support:>10.4f} {avg_f1/total_support:>10.4f} "
          f"{total_support:>10d}")


# ============================================================
# 🔍 NORMALISATION DES DONNÉES
# ============================================================

def min_max_normalize(X):
    """
    Normalisation Min-Max : ramener toutes les features entre 0 et 1.
    
    Formule : X_norm = (X - X_min) / (X_max - X_min)
    
    Pourquoi normaliser pour KNN ?
    ────────────────────────────────
    KNN utilise la distance. Si une feature a des valeurs de 0 à 1000
    et une autre de 0 à 1, la première dominera le calcul de distance !
    
    Exemple :
        Surface : [50, 100, 200] m²
        Chambres : [1, 2, 3]
        
        Sans normalisation : la distance sera dominée par Surface
        Avec normalisation : les deux features ont le même poids
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    
    # Éviter la division par zéro
    range_vals = X_max - X_min
    range_vals[range_vals == 0] = 1
    
    return (X - X_min) / range_vals


def z_score_normalize(X):
    """
    Normalisation Z-Score (Standardisation) : moyenne=0, écart-type=1.
    
    Formule : X_std = (X - μ) / σ
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std
