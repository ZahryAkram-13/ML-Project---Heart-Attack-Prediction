# main.py

import training_models  # On importe le fichier qui contient la logique d'entraînement

if __name__ == "__main__":
    print("Démarrage de l'entraînement du modèle...")
    #print(training_models.test())
    training_models.train()  # Appelle la fonction d'entraînement du modèle
