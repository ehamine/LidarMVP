# État du Projet LidarManagerV2

## Phase 0 : Initialisation du Projet (Terminée)

*   **Objectif :** Mettre en place un projet C++ de base, compilable, avec gestion des dépendances.
*   **Statut :** **Terminée**
*   **Livrables :**
    *   Projet CMake fonctionnel.
    *   Intégration de Conan pour la gestion des dépendances (`fmt`, `gtest`).
    *   Compilation et exécution d'un programme de test réussies.
*   **Notes :** L'intégration initiale de Conan via `cmake/conan_setup.cmake` était défectueuse. Le script a été désactivé au profit d'une approche standard plus robuste (`conan install` suivi de `cmake`).

## Phase 1 : Acquisition et Traitement de base du Lidar

*   **Objectif :** Mettre en place l'acquisition des données d'un capteur Ouster et définir les structures de base pour le traitement.
*   **Prochaines Étapes :**
    1.  **Ajouter la dépendance Ouster SDK :**
        *   Ajouter la dépendance pour le SDK Ouster au fichier `conanfile.txt`.
    2.  **Créer le module `LidarAcquisition` :**
        *   Mettre en place une classe qui utilise le SDK Ouster pour se connecter au capteur (ou lire un fichier PCAP) et récupérer les scans Lidar.
    3.  **Définir les structures de données :**
        *   Créer les en-têtes pour les types de points (ex: `PointXYZIRT`) et la structure du nuage de points, en s'inspirant de la spécification.
    4.  **Mettre à jour la fonction `main` :**
        *   Instancier le module d'acquisition.
        *   Mettre en place une boucle simple qui récupère les scans et affiche des informations de base (ex: nombre de points, timestamp).