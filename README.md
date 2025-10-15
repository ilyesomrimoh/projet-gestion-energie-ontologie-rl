
# ⚡ Gestion d’énergie avec Apprentissage par Renforcement (Q-Learning)

## 🧠 Description du projet
Ce projet met en œuvre un **système d’apprentissage par renforcement (RL)** simple pour optimiser l’utilisation de deux batteries alimentant un moteur électrique.  
L’objectif est d’apprendre une **politique de sélection de batterie** (A ou B) qui maximise la récompense en maintenant un équilibre entre leurs niveaux de charge tout en répondant à la demande énergétique du moteur.

Le projet combine :
- **Une ontologie RDF (Web sémantique)** pour représenter les composants énergétiques (batteries, moteur).
- **Un environnement RL personnalisé** (`EnergyEnv`) basé sur cette ontologie.
- **Un agent d’apprentissage Q-Learning** implémenté avec un **réseau de neurones (PyTorch)**.

---

## 🗂️ Structure du projet

```
├── EnergyEnv.py         # Environnement RL basé sur l’ontologie
├── ontology.py          # Création et gestion de l’ontologie RDF
├── main.py              # Entraînement du modèle avec Q-Learning
├── data/
│   └── ontology.ttl     # Fichier généré contenant l’ontologie RDF
└── README.md            # Description du projet
```

---

## ⚙️ Fonctionnement

### 1️⃣ Ontologie RDF
Le fichier `ontology.py` définit une ontologie simple :
- Deux batteries (`BatteryA` et `BatteryB`)
- Un moteur (`Motor`)
- Des propriétés :
  - `hasCharge` → niveau de charge de chaque batterie
  - `powerDemand` → demande du moteur

```python
BatteryA : hasCharge = 80
BatteryB : hasCharge = 40
Motor    : powerDemand = 50
```

Le graphe RDF est sauvegardé sous `data/ontology.ttl`.

---

### 2️⃣ Environnement RL — `EnergyEnv`
L’environnement lit les données depuis l’ontologie RDF et définit :
- **État (state)** : `[chargeA, chargeB]`
- **Actions (action)** :
  - `0` → utiliser la batterie A
  - `1` → utiliser la batterie B
- **Récompense (reward)** :
  - +10 points pour une action valide  
  - pénalité proportionnelle à la différence de charge entre les batteries  
  - -10 si une batterie tombe à zéro

Le but de l’agent est d’équilibrer les charges pour éviter qu’une batterie ne soit totalement déchargée.

---

### 3️⃣ Agent RL — `main.py`
Le fichier `main.py` implémente un agent **Q-Learning** à l’aide d’un petit **réseau de neurones** :

- **Entrée :** 2 (charges des batteries)  
- **Sortie :** 2 (actions possibles)
- **Réseau :**
  ```python
  nn.Linear(2, 32) → ReLU → nn.Linear(32, 2)
  ```
- **Optimiseur :** Adam (lr = 0.01)
- **Fonction de perte :** MSE
- **Facteur de discount :** γ = 0.9
- **Exploration :** ε-décroissant (de 0.1 à 0.01)

L’agent choisit quelle batterie utiliser pour maximiser la récompense cumulée.

---

## 📊 Résultats et Observations

- Le graphique final montre l’évolution du **total reward** au fil des épisodes.
- Le modèle apprend progressivement à **équilibrer l’utilisation des deux batteries**.
- Une **décroissance de ε** permet à l’agent de passer de l’exploration à l’exploitation.
- Si une batterie tombe à zéro, la pénalité de -10 encourage une gestion plus équilibrée.
- Le modèle entraîné est sauvegardé dans `qnet_model.pth` pour un futur réentraînement.

---

## 🚀 Lancer le projet

### 1️⃣ Installer les dépendances
```bash
pip install -r requirements.txt 
```

### 2️⃣ Générer l’ontologie
```bash
python ontology.py
```

### 3️⃣ Lancer l’entraînement
```bash
python main.py
```

---

## 🧩 Technologies utilisées
- **Python**
- **PyTorch** — pour le réseau de neurones du Q-Learning
- **RDFLib** — pour manipuler l’ontologie
- **Matplotlib** — pour visualiser la courbe de récompenses
- **NumPy** — pour gérer les états numériques

---

## 💡 Améliorations possibles
- Ajouter un **système de recharge automatique** des batteries.
- Utiliser **DQN (Deep Q-Network)** avec mémoire d’expérience (replay buffer).
- Introduire plus de variables dans l’ontologie (température, rendement, etc.).
- Comparer les performances entre **Q-Learning tabulaire** et **réseau de neurones**.
