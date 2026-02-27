
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


def train_test_split_manual_p(X, y, test_size=0.2, random_state=None):
    """
    Diviser les données en ensembles d'entraînement et de test.
    
    Paramètres :
        X : features
        y : labels
        test_size : proportion des données à mettre dans le jeu de test
        random_state : graine pour la reproductibilité
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(len(X))
    test_samples = int(n_samples*test_size)

    train_indices = indices[test_samples:]
    test_indices = indices[:test_samples]

    
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]



# ============================================================
# 🔍 NORMALISATION DES DONNÉES
# ============================================================

def z_score_normalize(X):
    """
    Normalisation Z-Score (Standardisation) : moyenne=0, écart-type=1.
    
    Formule : X_std = (X - μ) / σ
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std



# ============================================================
# 🧠 CLASSE KNN — L'ALGORITHME COMPLET
# ============================================================

class KN_Classifier:
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
        print("Le model a bien ete entraine !!")
        return self  # Pour permettre le chaînage : model.fit(X, y).predict(X_test)
    
    def predict_single(self, x: np.ndarray):
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
        k_voisins_labels = [self.y_train[i].item() if hasattr(self.y_train[i],'item') else self.y_train[i] for i in k_indices]
        
        # Étape 4 : Vote majoritaire
        # Counter({'Iris-setosa': 3, 'Iris-versicolor': 2}) → 'Iris-setosa'
        vote = Counter(k_voisins_labels)
    
        classe_predite = vote.most_common(1)[0][0]

        return classe_predite


    def predict(self, X: np.ndarray):
        """
        Prédire les classes pour un ensemble de points.
        
        Paramètres :
            X : np.ndarray de shape (n_samples, n_features)
        
        Retourne :
            np.ndarray de shape (n_samples,) avec les classes prédites
        """
        if self.X_train is None:
            raise RuntimeError("Le modèle n'a pas été entraîné ! Appelle fit() d'abord.")
        X = np.array(X)
        predictions = []
        for x in X :
            predictions.append(self.predict_single(x))

        return np.array(predictions)
            
    def predict_proba(self, X: np.ndarray) -> dict:
        """
        Renvoie les probabilités de chaque classe pour chaque point.
        
        La "probabilité" = proportion des K voisins appartenant à chaque classe.
        Exemple avec k=5 : si 3 voisins sont Setosa et 2 sont Versicolor,
                           → P(Setosa) = 3/5 = 0.6, P(Versicolor) = 2/5 = 0.4
        """
        all_probas = []
        X = np.array(X)
        
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
        y = np.array(y)
        accuracy = np.sum(predictions == np.array(y)) / len(y)
        return accuracy
    
    def __repr__(self):
        return f"KNN(k={self.k}, metric='{self.metric}')"


# ============================================================
# 📊 FONCTIONS D'ÉVALUATION
# ============================================================

def accuracy_score(y_true, y_pred):
    """
    Calculer l'accuracy (précision globale).
    
    Accuracy = (nombre de prédictions correctes) / (nombre total de prédictions)
    """
    return np.mean(np.array(y_true) == np.array(y_pred))



# ============================================================
# 🔍 K-NEAREST NEIGHBORS REGRESSOR (REGRESSION)
# ============================================================


class KN_Regressor:
    """
    K-Nearest Neighbors Regressor — implémenté from scratch.
    
    Fonctionnement :
    1. Pour chaque point de test, trouver les K voisins les plus proches
    2. Calculer la moyenne des valeurs des voisins
    3. Retourner cette moyenne comme prédiction
    """
    
    def __init__(self, k=5, metric='euclidean'):
        """
        Initialiser le régresseur KNN.
        
        Paramètres :
            k : int, nombre de voisins à considérer
            metric : str, métrique de distance ('euclidean' ou 'manhattan')
        """
        self.k = k
        self.metric = metric
        
        # Choisir la fonction de distance
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
        
        Paramètres :
            X : np.ndarray de shape (n_samples, n_features)
            y : np.ndarray de shape (n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).ravel()  # Toujours 1D pour éviter les problèmes de broadcasting
        
        if self.k > len(self.X_train):
            raise ValueError(
                f"k={self.k} est plus grand que le nombre "
                f"d'échantillons d'entraînement ({len(self.X_train)})"
            )
        
        return self
    
    def _predict_single(self, x: np.ndarray) -> float:
        """
        Prédire la valeur pour UN SEUL point.
        
        Étapes :
        1. Calculer la distance entre x et tous les points d'entraînement
        2. Trier par distance croissante
        3. Prendre les K plus proches
        4. Calculer la moyenne de leurs valeurs
        """
        x = np.array(x)
        # Étape 1 : Calculer toutes les distances
        distances = np.array([
            self._distance_fn(x, x_train)
            for x_train in self.X_train
        ])
        
        # Étape 2 : Obtenir les indices des K plus petites distances
        k_indices = np.argsort(distances)[:self.k]
        
        # Étape 3 : Récupérer les valeurs des K plus proches voisins
        k_voisins_values = self.y_train[k_indices]
        
        # Étape 4 : Calculer la moyenne (régression)
        prediction = np.mean(k_voisins_values)
        
        return prediction
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédire les valeurs pour un ensemble de points.
        
        Paramètres :
            X : np.ndarray de shape (n_samples, n_features)
        
        Retourne :
            np.ndarray de shape (n_samples,) avec les valeurs prédites
        """
        X = np.array(X)
        if self.X_train is None:
            raise RuntimeError("Le modèle n'a pas été entraîné ! Appelle fit() d'abord.")
        
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculer le score R² (coefficient de détermination).
        
        R² = 1 - (SS_res / SS_tot)
           = 1 - (somme des carrés des résidus / somme des carrés totaux)
        
        - 1.0 = prédictions parfaites
        - 0.0 = aussi bon que de prédire la moyenne
        - < 0 = pire que de prédire la moyenne
        """
        X = np.array(X)
        y = np.array(y).ravel()  # Toujours 1D pour éviter les problèmes de broadcasting
        predictions = self.predict(X)
        
        # Calculer SS_res (somme des carrés des résidus)
        somme_residus = np.sum((y - predictions) ** 2)
        
        # Calculer SS_tot (somme des carrés totaux)
        somme_totale = np.sum((y - np.mean(y)) ** 2)
        
        # Éviter la division par zéro
        if somme_totale == 0:
            return 1.0 if somme_residus == 0 else 0.0
        
        r2_score = 1 - (somme_residus / somme_totale)
        return r2_score

    def __repr__(self):
        return f"KNNRegressor(k={self.k}, metric='{self.metric}')"
     
# Fonction train test split manual pour datasets non pandas (Non Dataframes)

def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test.
    """
    # Fixer la seed pour la reproductibilité
    if random_state is not None:
        np.random.seed(random_state)
    
    # Conversion en numpy array pour assurer le bon fonctionnement
    X = np.array(X)
    y = np.array(y)
    
    # 1. Créer et mélanger les indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # 2. Calculer le point de coupure
    test_set_size = int(len(X) * test_size)
    
    # 3. Séparer les indices
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    # 4. Créer les sous-ensembles
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test
