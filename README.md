# Q-Learning Battery Charging Agent 🔋🤖

Ce projet implémente un **agent d'apprentissage par renforcement (Q-learning)** qui apprend à **gérer la charge d'une batterie** de manière optimale.  
L'objectif est de maximiser la récompense totale tout en maintenant un comportement efficace à long terme.

---

## 🚀 Objectif du projet

Le but de ce projet est de **simuler un environnement de charge de batterie** dans lequel un agent intelligent apprend à décider :
- Quand charger la batterie.
- Quand économiser l'énergie.
- Comment maximiser la performance sur plusieurs épisodes.

---

## 🧠 Fonctionnement

L'agent apprend grâce à une approche **Q-Learning avec réseau de neurones (Q-Network)**.

### 1. Environnement (`BatteryEnv`)
L’environnement simule une batterie avec un niveau de charge (entre 0 et 100).  
Chaque action (charger ou ne rien faire) modifie ce niveau de charge et donne une **récompense** selon le comportement.

- **État (state)** : Niveau de charge de la batterie.
- **Actions (action)** :
  - `0` → Ne rien faire.
  - `1` → Charger.
- **Récompense (reward)** :
  - Positive si l’action est efficace (ex : maintenir la batterie dans une plage optimale).
  - Négative si l’action est inefficace (ex : surcharge ou sous-charge).

### 2. Agent Q-Learning (`qnet`)
Le réseau de neurones `qnet` estime les valeurs Q (Q-values) pour chaque action possible dans un état donné.

Formule de mise à jour :
```
Q(s, a) ← Q(s, a) + α [r + γ * max(Q(s', a')) − Q(s, a)]
```
- `α` : Taux d’apprentissage (learning rate).
- `γ` : Facteur de récompense future (discount factor).

### 3. Entraînement
Le fichier `train_qlearning()` entraîne l’agent sur plusieurs épisodes.  
L’agent explore d’abord (grâce à `epsilon`), puis exploite ce qu’il a appris (réduction progressive d’`epsilon`).

---

## 📈 Visualisation des résultats

À la fin de l’entraînement, une courbe montre la **récompense totale par épisode**, permettant d’observer la progression de l’agent.

```python
plt.plot(range(1, len(rewards) + 1), rewards)
plt.xlabel("Épisodes")
plt.ylabel("Récompense totale")
plt.title("Progression de l'apprentissage")
plt.show()
```

---

## 🧩 Structure du projet

```
📂 QLearning-Battery
├── qnet_model.pth          # Poids du modèle sauvegardé
├── main.py                 # Point d'entrée principal
├── train.py                # Fonction d'entraînement Q-learning
├── env.py                  # Définition de l'environnement BatteryEnv
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier 📘
```

---

## 🛠️ Technologies utilisées

- **Python 3**
- **PyTorch** (réseau de neurones)
- **Matplotlib** (visualisation)
- **Numpy**

---

## 📊 Observations

- Le modèle apprend à stabiliser sa stratégie de charge après plusieurs épisodes.  
- Il peut parfois présenter des **fluctuations** (récompenses en baisse temporaire) dues à :
  - L’exploration (`epsilon`).
  - La nature stochastique de l’environnement.
- Plus le nombre d’épisodes augmente, plus le comportement devient stable.

---

## 🔄 Améliorations possibles

- Ajouter une mémoire de rejouage (Experience Replay).
- Implémenter une version **Deep Q-Learning (DQN)**.
- Introduire un **epsilon dynamique** pour équilibrer exploration/exploitation.
- Simuler plusieurs batteries ou différents environnements de charge.

---

## 👨‍💻 Auteur
Projet réalisé par **Ilyes Omri**, étudiant en informatique passionné par l’intelligence artificielle et l’apprentissage par renforcement.

