# ReturnSense — Système de Prédiction de Retours

## Structure
```
return_predictor/
├── app.py                  ← Backend Flask
├── requirements.txt        ← Dépendances Python
├── best_return_model.pkl   ← ⚠ À placer ici (votre modèle entraîné)
└── templates/
    └── index.html          ← Interface web
```

## Installation & Lancement

```bash
# 1. lancer cette commande dans le terminal
make run

# 4. Ouvrir dans le navigateur
# http://localhost:5000
```

## Notes importantes

- Le modèle doit être sauvegardé avec `pickle.dump(model, f)`
- Le seuil de décision optimal est fixé à **0.31**
- Le pipeline de features dans `app.py` doit correspondre exactement
  aux features utilisées lors de l'entraînement
- Si votre modèle a des features supplémentaires, ajoutez-les dans
  la fonction `build_features()` dans `app.py`# PFE
