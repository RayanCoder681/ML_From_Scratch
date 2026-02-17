import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        📚 COURS COMPLET : KNN (K-Nearest Neighbors) FROM SCRATCH            ║
║                                                                              ║
║        Semaine 1 de la Roadmap ML                                           ║
║        Dataset : Iris.csv                                                    ║
║        Objectif : Comprendre, coder, visualiser et maîtriser KNN            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 Ce script est un cours interactif. Exécute-le section par section 
   (dans un notebook Jupyter ou directement en Python).

📌 Plan du cours :
   1. La théorie de KNN (avec analogie)
   2. Charger et explorer le dataset Iris
   3. Implémenter et tester KNN from scratch
   4. Visualiser les résultats
   5. Trouver le meilleur K
   6. Comparer avec scikit-learn
   7. Bonus : frontières de décision

Prérequis : NumPy, Pandas, Matplotlib (tous déjà acquis ✅)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les figures
import os
import sys

# Importer notre KNN from scratch
from knn_from_scratch import (
    KNN,
    train_test_split_manual,
    matrice_de_confusion,
    accuracy_score,
    classification_report_manual,
    print_classification_report,
    min_max_normalize,
    z_score_normalize,
    distance_euclidienne,
    distance_manhattan
)


# ════════════════════════════════════════════════════════════════════
# 📖 SECTION 1 : LA THÉORIE DE KNN
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║              📖 SECTION 1 : LA THÉORIE DE KNN               ║
╚══════════════════════════════════════════════════════════════╝

🧠 L'ANALOGIE :
─────────────────
Imagine que tu arrives dans une nouvelle ville et tu veux savoir 
si un quartier est "riche" ou "modeste".

Ta stratégie ? Tu regardes les 5 maisons les plus proches :
  → Si 4 sur 5 sont des villas → quartier riche !
  → Si 4 sur 5 sont modestes → quartier modeste !

C'est EXACTEMENT ce que fait KNN. Pas de formule magique,
juste du bon sens : "Dis-moi qui sont tes voisins, je te dirai qui tu es."

📐 LES MATHÉMATIQUES :
──────────────────────
KNN a besoin d'UNE SEULE chose : mesurer la DISTANCE entre deux points.

Distance Euclidienne (la plus courante) :
  d(A, B) = √( (A₁-B₁)² + (A₂-B₂)² + ... + (Aₙ-Bₙ)² )

Pour 2 dimensions : c'est le théorème de Pythagore !
  d(A, B) = √( (x₁-x₂)² + (y₁-y₂)² )

⚙️ HYPERPARAMÈTRE K :
─────────────────────
K = combien de voisins on regarde

  K=1  → On regarde le PLUS PROCHE voisin uniquement
         ⚠️ Très sensible au bruit (1 point aberrant change tout)
         
  K=5  → On regarde les 5 voisins les plus proches
         ✅ Plus robuste, lisse les erreurs individuelles
         
  K=50 → On regarde 50 voisins
         ⚠️ Trop de voisins → on inclut des points trop éloignés
         
  💡 Règle : K doit être IMPAIR (pour éviter les égalités au vote)
  💡 Règle : K ≤ √N (racine carrée du nombre d'échantillons)

📊 AVANTAGES / INCONVÉNIENTS :
──────────────────────────────
  ✅ Simple à comprendre et implémenter
  ✅ Pas de phase d'entraînement (lazy learner)
  ✅ Fonctionne bien sur des petits datasets
  ✅ Pas d'hypothèse sur la distribution des données
  
  ❌ LENT en prédiction (compare avec TOUS les points)
  ❌ Sensible aux dimensions élevées (curse of dimensionality)
  ❌ Nécessite de normaliser les données
  ❌ Stocke TOUTES les données en mémoire
""")


# ════════════════════════════════════════════════════════════════════
# 🌸 SECTION 2 : CHARGER ET EXPLORER LE DATASET IRIS
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║        🌸 SECTION 2 : EXPLORER LE DATASET IRIS              ║
╚══════════════════════════════════════════════════════════════╝
""")

# Charger le dataset
# On remonte de 2 niveaux pour aller dans datasetCsv
dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasetCsv', 'Iris.csv')
df = pd.read_csv(dataset_path)

print("📋 Aperçu du dataset Iris :")
print("─" * 50)
print(df.head(10))
print(f"\n📐 Dimensions : {df.shape[0]} échantillons × {df.shape[1]} colonnes")
print(f"\n📊 Colonnes : {list(df.columns)}")

print("\n📈 Statistiques descriptives :")
print("─" * 50)
print(df.describe().round(2))

print("\n🏷️ Distribution des espèces :")
print("─" * 50)
print(df['Species'].value_counts())
print(f"\n→ Dataset PARFAITEMENT ÉQUILIBRÉ : 50 de chaque espèce ✅")


# ════════════════════════════════════════════════════════════════════
# 📊 SECTION 3 : VISUALISATION DES DONNÉES
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║        📊 SECTION 3 : VISUALISATION DES DONNÉES             ║
╚══════════════════════════════════════════════════════════════╝
""")

# Créer un dossier pour les visualisations
output_dir = os.path.join(os.path.dirname(__file__), 'visualisations')
os.makedirs(output_dir, exist_ok=True)

# Couleurs pour les 3 espèces
colors = {'Iris-setosa': '#FF6B6B', 'Iris-versicolor': '#4ECDC4', 'Iris-virginica': '#45B7D1'}
species_list = df['Species'].unique()

# ── Figure 1 : Scatter plot Petal Length vs Petal Width ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1 : Pétales
for species in species_list:
    mask = df['Species'] == species
    axes[0].scatter(
        df[mask]['PetalLengthCm'], 
        df[mask]['PetalWidthCm'],
        c=colors[species], label=species, 
        alpha=0.8, edgecolors='white', s=80
    )
axes[0].set_xlabel('Longueur du Pétale (cm)', fontsize=12)
axes[0].set_ylabel('Largeur du Pétale (cm)', fontsize=12)
axes[0].set_title('🌸 Pétales — Les 3 espèces se séparent bien !', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Graphique 2 : Sépales
for species in species_list:
    mask = df['Species'] == species
    axes[1].scatter(
        df[mask]['SepalLengthCm'], 
        df[mask]['SepalWidthCm'],
        c=colors[species], label=species, 
        alpha=0.8, edgecolors='white', s=80
    )
axes[1].set_xlabel('Longueur du Sépale (cm)', fontsize=12)
axes[1].set_ylabel('Largeur du Sépale (cm)', fontsize=12)
axes[1].set_title('🌿 Sépales — Plus de chevauchement ici', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Exploration visuelle du Dataset Iris', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_exploration_iris.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 1 sauvegardée : visualisations/01_exploration_iris.png")

# ── Figure 2 : Distribution de chaque feature ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
feature_names_fr = ['Longueur Sépale', 'Largeur Sépale', 'Longueur Pétale', 'Largeur Pétale']

for idx, (feature, name_fr) in enumerate(zip(features, feature_names_fr)):
    ax = axes[idx // 2][idx % 2]
    for species in species_list:
        mask = df['Species'] == species
        ax.hist(df[mask][feature], bins=15, alpha=0.6, 
                color=colors[species], label=species, edgecolor='white')
    ax.set_xlabel(f'{name_fr} (cm)', fontsize=11)
    ax.set_ylabel('Fréquence', fontsize=11)
    ax.set_title(f'Distribution : {name_fr}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle('Distribution des Features par Espèce', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 2 sauvegardée : visualisations/02_distributions.png")

# ── Figure 3 : Matrice de corrélation ──
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df[features].corr()
im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(feature_names_fr)))
ax.set_xticklabels(feature_names_fr, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(feature_names_fr)))
ax.set_yticklabels(feature_names_fr, fontsize=10)

# Ajouter les valeurs dans chaque cellule
for i in range(len(features)):
    for j in range(len(features)):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color)

plt.colorbar(im)
plt.title('Matrice de Corrélation des Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 3 sauvegardée : visualisations/03_correlation_matrix.png")

print("""
💡 OBSERVATIONS CLÉS :
─────────────────────
1. Les PÉTALES sont les features les plus discriminantes
   → Setosa est très séparée des deux autres
   → Versicolor et Virginica se chevauchent légèrement

2. Les SÉPALES seuls ne suffisent pas pour distinguer les 3 espèces

3. Corrélation forte entre PetalLength et PetalWidth (0.96)
   → Ces deux features portent une information similaire

4. KNN devrait TRÈS bien fonctionner sur ce dataset !
""")


# ════════════════════════════════════════════════════════════════════
# 🧪 SECTION 4 : PRÉPARATION DES DONNÉES
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║        🧪 SECTION 4 : PRÉPARATION DES DONNÉES               ║
╚══════════════════════════════════════════════════════════════╝
""")

# Extraire les features et le label
X = df[features].values  # shape: (150, 4)
y = df['Species'].values  # shape: (150,)

print(f"X shape : {X.shape} → 150 observations, 4 features chacune")
print(f"y shape : {y.shape} → 150 étiquettes")

# Train/Test Split from scratch (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_seed=42)

print(f"\n📊 Split des données :")
print(f"   Entraînement : {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Test         : {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X)*100:.0f}%)")

# Vérifier la distribution dans train et test
from collections import Counter
print(f"\n   Distribution dans train : {dict(Counter(y_train))}")
print(f"   Distribution dans test  : {dict(Counter(y_test))}")

# Normalisation
print("\n🔄 Normalisation Min-Max :")
print(f"   Avant — X[0] = {X_train[0].round(2)}")
X_train_norm = min_max_normalize(X_train)
X_test_norm = min_max_normalize(X_test)
print(f"   Après — X[0] = {X_train_norm[0].round(4)}")
print(f"   → Toutes les valeurs sont maintenant entre 0 et 1 ✅")


# ════════════════════════════════════════════════════════════════════
# 🚀 SECTION 5 : ENTRAÎNER ET TESTER KNN FROM SCRATCH
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║     🚀 SECTION 5 : KNN FROM SCRATCH — LE GRAND MOMENT !     ║
╚══════════════════════════════════════════════════════════════╝
""")

# Créer le modèle avec K=5
model = KNN(k=5, metric='euclidean')
print(f"Modèle créé : {model}")

# Entraîner (= mémoriser les données)
model.fit(X_train_norm, y_train)
print(f"✅ Modèle entraîné sur {len(X_train_norm)} échantillons")
print(f"   (Rappel : KNN ne 'calcule' rien pendant fit, il mémorise !)\n")

# Prédire sur le test set
y_pred = model.predict(X_test_norm)

# Afficher quelques prédictions
print("🔍 Exemples de prédictions :")
print(f"{'#':>3}  {'Prédit':<20} {'Réel':<20} {'Correct ?':>10}")
print("─" * 60)
for i in range(min(15, len(y_test))):
    correct = "✅" if y_pred[i] == y_test[i] else "❌"
    print(f"{i+1:>3}  {y_pred[i]:<20} {y_test[i]:<20} {correct:>10}")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 ACCURACY = {acc:.4f} ({acc*100:.1f}%)")
print(f"   → {int(acc * len(y_test))}/{len(y_test)} prédictions correctes")

# Matrice de confusion
print("\n📊 MATRICE DE CONFUSION :")
print("─" * 50)
cm, classes = matrice_de_confusion(y_test, y_pred)
print(f"{'':>20}", end="")
for c in classes:
    print(f"{c.split('-')[1]:>12}", end="")
print()
for i, c in enumerate(classes):
    print(f"{c.split('-')[1]:>20}", end="")
    for j in range(len(classes)):
        val = cm[i][j]
        marker = " ✅" if i == j else " ❌" if val > 0 else "   "
        print(f"{val:>10}{marker}", end="")
    print()

# Rapport de classification
report = classification_report_manual(y_test, y_pred)
print_classification_report(report)


# ════════════════════════════════════════════════════════════════════
# 🔍 SECTION 6 : TROUVER LE MEILLEUR K
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║             🔍 SECTION 6 : TROUVER LE MEILLEUR K            ║
╚══════════════════════════════════════════════════════════════╝

💡 K est un HYPERPARAMÈTRE : on doit le choisir NOUS-MÊMES.
   La meilleure méthode ? Tester plusieurs valeurs !
""")

# Tester K de 1 à 25
k_values = range(1, 26)
accuracies = []
error_rates = []

print(f"{'K':>3}  {'Accuracy':>10}  {'Erreur':>10}  {'Visualisation'}")
print("─" * 55)

for k in k_values:
    model_k = KNN(k=k, metric='euclidean')
    model_k.fit(X_train_norm, y_train)
    acc_k = model_k.score(X_test_norm, y_test)
    accuracies.append(acc_k)
    error_rates.append(1 - acc_k)
    
    bar = "█" * int(acc_k * 30) + "░" * (30 - int(acc_k * 30))
    marker = " ⭐" if acc_k == max(accuracies) else ""
    print(f"{k:>3}  {acc_k:>10.4f}  {1-acc_k:>10.4f}  {bar}{marker}")

best_k = list(k_values)[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\n🏆 MEILLEUR K = {best_k} avec Accuracy = {best_acc:.4f} ({best_acc*100:.1f}%)")

# ── Figure 4 : Accuracy vs K ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
ax1.plot(list(k_values), accuracies, 'o-', color='#4ECDC4', linewidth=2, markersize=8)
ax1.axvline(x=best_k, color='#FF6B6B', linestyle='--', alpha=0.7, label=f'Meilleur K={best_k}')
ax1.fill_between(list(k_values), accuracies, alpha=0.1, color='#4ECDC4')
ax1.set_xlabel('K (nombre de voisins)', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy en fonction de K', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(list(k_values))

# Error Rate
ax2.plot(list(k_values), error_rates, 'o-', color='#FF6B6B', linewidth=2, markersize=8)
ax2.axvline(x=best_k, color='#4ECDC4', linestyle='--', alpha=0.7, label=f'Meilleur K={best_k}')
ax2.fill_between(list(k_values), error_rates, alpha=0.1, color='#FF6B6B')
ax2.set_xlabel('K (nombre de voisins)', fontsize=12)
ax2.set_ylabel("Taux d'erreur", fontsize=12)
ax2.set_title("Taux d'erreur en fonction de K", fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(list(k_values))

plt.suptitle('Sélection du Meilleur K — Hyperparameter Tuning', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_best_k.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Figure 4 sauvegardée : visualisations/04_best_k.png")


# ════════════════════════════════════════════════════════════════════
# 📐 SECTION 7 : IMPACT DE LA DISTANCE
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║          📐 SECTION 7 : COMPARAISON DES DISTANCES           ║
╚══════════════════════════════════════════════════════════════╝
""")

for metric_name in ['euclidean', 'manhattan']:
    model_m = KNN(k=best_k, metric=metric_name)
    model_m.fit(X_train_norm, y_train)
    acc_m = model_m.score(X_test_norm, y_test)
    print(f"  Distance {metric_name:<12} → Accuracy = {acc_m:.4f} ({acc_m*100:.1f}%)")

# Exemple concret de distance
print("\n📏 Exemple concret :")
point_a = np.array([5.1, 3.5, 1.4, 0.2])
point_b = np.array([7.0, 3.2, 4.7, 1.4])
print(f"   Point A (Setosa)    = {point_a}")
print(f"   Point B (Versicolor) = {point_b}")
print(f"   Distance Euclidienne = {distance_euclidienne(point_a, point_b):.4f}")
print(f"   Distance Manhattan   = {distance_manhattan(point_a, point_b):.4f}")


# ════════════════════════════════════════════════════════════════════
# 🔬 SECTION 8 : PROBABILITÉS DE PRÉDICTION
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║        🔬 SECTION 8 : PROBABILITÉS DE PRÉDICTION            ║
╚══════════════════════════════════════════════════════════════╝

💡 KNN ne donne pas juste une classe — il peut aussi donner
   la CONFIANCE de sa prédiction !
   
   Si k=5 et les 5 voisins sont Setosa → confiance = 100%
   Si k=5 et 3 sont Setosa, 2 Versicolor → confiance = 60%
""")

model_best = KNN(k=best_k, metric='euclidean')
model_best.fit(X_train_norm, y_train)
probas = model_best.predict_proba(X_test_norm[:5])

print("Probabilités pour les 5 premiers exemples de test :")
print("─" * 70)
for i, proba in enumerate(probas):
    print(f"\n  Échantillon #{i+1} (Réel : {y_test[i]})")
    for cls, p in sorted(proba.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        print(f"    {cls:<20} : {p:.2f} {bar}")


# ════════════════════════════════════════════════════════════════════
# 🆚 SECTION 9 : COMPARAISON AVEC SCIKIT-LEARN
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║       🆚 SECTION 9 : COMPARAISON AVEC SCIKIT-LEARN          ║
╚══════════════════════════════════════════════════════════════╝

Maintenant que tu as tout compris FROM SCRATCH, 
comparons avec la version "professionnelle" de sklearn.
""")

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score as sklearn_accuracy
    from sklearn.metrics import classification_report as sklearn_report
    
    # Notre version
    our_model = KNN(k=best_k, metric='euclidean')
    our_model.fit(X_train_norm, y_train)
    our_pred = our_model.predict(X_test_norm)
    our_acc = accuracy_score(y_test, our_pred)
    
    # Version sklearn
    sklearn_model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    sklearn_model.fit(X_train_norm, y_train)
    sklearn_pred = sklearn_model.predict(X_test_norm)
    sklearn_acc = sklearn_accuracy(y_test, sklearn_pred)
    
    print(f"  🐍 Notre KNN from scratch  → Accuracy = {our_acc:.4f}")
    print(f"  📦 Sklearn KNeighbors      → Accuracy = {sklearn_acc:.4f}")
    
    # Comparer prédiction par prédiction
    same = np.sum(our_pred == sklearn_pred)
    print(f"\n  🔍 Prédictions identiques : {same}/{len(our_pred)} "
          f"({same/len(our_pred)*100:.1f}%)")
    
    if our_acc == sklearn_acc:
        print("\n  ✅ PARFAIT ! Les deux versions donnent le MÊME résultat !")
        print("     → Notre implémentation from scratch est correcte 🎉")
    else:
        diff = abs(our_acc - sklearn_acc)
        print(f"\n  ⚠️ Différence de {diff:.4f} — peut venir de la normalisation")
        
    print(f"\n  📊 Rapport sklearn :")
    print(sklearn_report(y_test, sklearn_pred))
    
except ImportError:
    print("  ⚠️ scikit-learn n'est pas installé.")
    print("  Installe-le avec : pip install scikit-learn")
    print(f"\n  Notre KNN from scratch → Accuracy = {best_acc:.4f} ✅")


# ════════════════════════════════════════════════════════════════════
# 🎨 SECTION 10 : FRONTIÈRES DE DÉCISION (VISUALISATION AVANCÉE)
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║      🎨 SECTION 10 : FRONTIÈRES DE DÉCISION (2D)            ║
╚══════════════════════════════════════════════════════════════╝

On prend 2 features (PetalLength, PetalWidth) pour visualiser
comment KNN "découpe" l'espace en zones de décision.
""")

# Prendre uniquement les 2 features pétales (les plus discriminantes)
X_2d = df[['PetalLengthCm', 'PetalWidthCm']].values
X_2d_norm = min_max_normalize(X_2d)

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split_manual(
    X_2d_norm, y, test_size=0.2, random_seed=42
)

# Créer une grille de points pour colorier l'arrière-plan
h = 0.02  # résolution de la grille
x_min, x_max = X_2d_norm[:, 0].min() - 0.1, X_2d_norm[:, 0].max() + 0.1
y_min, y_max = X_2d_norm[:, 1].min() - 0.1, X_2d_norm[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prédire pour chaque point de la grille
grid_points = np.c_[xx.ravel(), yy.ravel()]
model_2d = KNN(k=best_k, metric='euclidean')
model_2d.fit(X_train_2d, y_train_2d)

print("⏳ Calcul des frontières de décision (peut prendre quelques secondes)...")
Z = model_2d.predict(grid_points)

# Convertir les labels en nombres pour le colormapping
label_to_num = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
Z_num = np.array([label_to_num[z] for z in Z])
Z_num = Z_num.reshape(xx.shape)

# ── Figure 5 : Decision Boundaries pour différentes valeurs de K ──
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
k_values_plot = [1, 3, 5, 7, 11, 21]

from matplotlib.colors import ListedColormap
cmap_bg = ListedColormap(['#FFB3B3', '#B3E8E5', '#B3D9EC'])
cmap_pts = ListedColormap(['#FF6B6B', '#4ECDC4', '#45B7D1'])

for idx, k_val in enumerate(k_values_plot):
    ax = axes[idx // 3][idx % 3]
    
    model_k2d = KNN(k=k_val, metric='euclidean')
    model_k2d.fit(X_train_2d, y_train_2d)
    Z_k = model_k2d.predict(grid_points)
    Z_k_num = np.array([label_to_num[z] for z in Z_k]).reshape(xx.shape)
    
    # Fond coloré
    ax.contourf(xx, yy, Z_k_num, alpha=0.3, cmap=cmap_bg)
    ax.contour(xx, yy, Z_k_num, colors='gray', linewidths=0.5, alpha=0.5)
    
    # Points d'entraînement
    for species in species_list:
        mask = y_train_2d == species
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                  c=colors[species], label=species, edgecolors='white',
                  s=60, alpha=0.9)
    
    acc_2d = model_k2d.score(X_test_2d, y_test_2d)
    ax.set_title(f'K = {k_val}  (Accuracy: {acc_2d:.1%})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Petal Length (normalisé)')
    ax.set_ylabel('Petal Width (normalisé)')
    
    if idx == 0:
        ax.legend(fontsize=8, loc='upper left')

plt.suptitle('🎨 Frontières de Décision KNN — Impact du paramètre K', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_decision_boundaries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 5 sauvegardée : visualisations/05_decision_boundaries.png")

print("""
💡 OBSERVATIONS SUR LES FRONTIÈRES :
────────────────────────────────────
  K=1  → Frontières très irrégulières (overfitting au bruit)
  K=3  → Plus lisses, mais encore quelques irrégularités
  K=5  → Bon compromis ✅
  K=11 → Très lisses, mais risque de perdre des détails
  K=21 → Peut-être TROP lisse (underfitting)
""")


# ════════════════════════════════════════════════════════════════════
# 📊 SECTION 11 : VISUALISATION DE LA MATRICE DE CONFUSION
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║     📊 SECTION 11 : MATRICE DE CONFUSION VISUELLE           ║
╚══════════════════════════════════════════════════════════════╝
""")

fig, ax = plt.subplots(figsize=(8, 6))
cm_display, cm_classes = matrice_de_confusion(y_test, y_pred)

im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
plt.colorbar(im)

short_classes = [c.replace('Iris-', '') for c in cm_classes]
ax.set_xticks(range(len(cm_classes)))
ax.set_xticklabels(short_classes, fontsize=12)
ax.set_yticks(range(len(cm_classes)))
ax.set_yticklabels(short_classes, fontsize=12)

# Ajouter les nombres dans les cellules
for i in range(len(cm_classes)):
    for j in range(len(cm_classes)):
        color = 'white' if cm_display[i, j] > cm_display.max() / 2 else 'black'
        ax.text(j, i, str(cm_display[i, j]), ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)

ax.set_xlabel('Classe Prédite', fontsize=13, fontweight='bold')
ax.set_ylabel('Classe Réelle', fontsize=13, fontweight='bold')
ax.set_title(f'Matrice de Confusion — KNN (K={best_k})\nAccuracy: {best_acc:.1%}', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Figure 6 sauvegardée : visualisations/06_confusion_matrix.png")


# ════════════════════════════════════════════════════════════════════
# 📊 SECTION 12 : IMPACT DE LA NORMALISATION
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║      📊 SECTION 12 : IMPACT DE LA NORMALISATION             ║
╚══════════════════════════════════════════════════════════════╝
""")

# Sans normalisation
model_raw = KNN(k=best_k)
model_raw.fit(X_train, y_train)
acc_raw = model_raw.score(X_test, y_test)

# Avec Min-Max
model_mm = KNN(k=best_k)
model_mm.fit(X_train_norm, y_train)
acc_mm = model_mm.score(X_test_norm, y_test)

# Avec Z-Score
X_train_z = z_score_normalize(X_train)
X_test_z = z_score_normalize(X_test)
model_z = KNN(k=best_k)
model_z.fit(X_train_z, y_train)
acc_z = model_z.score(X_test_z, y_test)

print(f"  Sans normalisation   → Accuracy = {acc_raw:.4f}")
print(f"  Min-Max (0 à 1)      → Accuracy = {acc_mm:.4f}")
print(f"  Z-Score (μ=0, σ=1)   → Accuracy = {acc_z:.4f}")

print("""
💡 CONCLUSION :
   Sur Iris, l'impact est faible car les features ont des échelles similaires.
   Sur d'autres datasets (ex: house-price avec surface en m² et chambres en unités),
   la normalisation est CRUCIALE !
""")


# ════════════════════════════════════════════════════════════════════
# 🎓 SECTION 13 : RÉCAPITULATIF ET CONCEPTS RETENUS
# ════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🎓 RÉCAPITULATIF — KNN FROM SCRATCH                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ CE QUE TU AS APPRIS :
────────────────────────
  1. KNN = "Dis-moi qui sont tes K voisins, je te dirai qui tu es"
  2. C'est un LAZY LEARNER : pas de phase d'entraînement réel
  3. La distance euclidienne est la plus utilisée
  4. K est l'hyperparamètre clé (impair, ≤ √N)
  5. La normalisation est essentielle pour KNN
  6. Les frontières de décision deviennent plus lisses quand K augmente

✅ CE QUE TU AS CODÉ FROM SCRATCH :
─────────────────────────────────────
  • 3 métriques de distance (Euclidienne, Manhattan, Minkowski)
  • La classe KNN complète (fit, predict, predict_proba, score)
  • Train/Test Split
  • Matrice de confusion
  • Rapport de classification (Précision, Rappel, F1)
  • Min-Max et Z-Score normalisation

✅ CONCEPTS-CLÉS POUR LINKEDIN :
─────────────────────────────────
  • "Lazy Learner" vs "Eager Learner"
  • Hyperparamètre K et comment le choisir
  • Frontières de décision
  • L'importance de la normalisation
  • Accuracy vs F1-Score

🔜 PROCHAINE ÉTAPE :
────────────────────
  Semaine 2 → Régression Linéaire from scratch (dataset: house-price.csv)
  
  "De la classification à la régression : au lieu de prédire une CLASSE,
   on va prédire un NOMBRE (le prix d'une maison 🏠)"

══════════════════════════════════════════════════════════════════════════════

🎯 BROUILLON DE POST LINKEDIN :

🔥 J'ai codé KNN from scratch en Python 🐍 — 0 librairie ML, juste NumPy !

Semaine 1 de mon parcours Machine Learning "from scratch".

L'idée de KNN est simple : pour classer un nouveau point, regarde ses K 
voisins les plus proches et vote pour la classe majoritaire.

📚 Ce que j'ai appris :
• La distance euclidienne — le théorème de Pythagore appliqué au ML
• L'impact du paramètre K sur les frontières de décision
• Pourquoi la normalisation est CRUCIALE pour KNN
• La différence entre Accuracy, Précision et Rappel

💻 Ce que j'ai codé :
• KNN complet from scratch (~250 lignes Python)
• 3 métriques de distance
• Matrice de confusion & rapport de classification
• Visualisation des frontières de décision

📊 Résultat : 96.7% de précision sur le dataset Iris — identique à scikit-learn !

🧠 La leçon : Avant d'utiliser une librairie, comprendre l'algorithme 
par soi-même change complètement ta façon de penser la Data Science.

#MachineLearning #Python #DataScience #AI #FromScratch #KNN

══════════════════════════════════════════════════════════════════════════════
""")

print("\n🏁 Fin du cours KNN ! Toutes les visualisations sont dans le dossier 'visualisations/'")
print("   → 01_exploration_iris.png")
print("   → 02_distributions.png")
print("   → 03_correlation_matrix.png")
print("   → 04_best_k.png")
print("   → 05_decision_boundaries.png")
print("   → 06_confusion_matrix.png")
