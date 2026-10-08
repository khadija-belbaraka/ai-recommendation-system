# AI Recommendation System

## Présentation du projet

Ce projet consiste à développer un système de recommandation intelligent capable de proposer des services personnalisés en fonction des caractéristiques des services et des interactions des utilisateurs.

Le système combine plusieurs approches de recommandation afin d'améliorer la pertinence des résultats et de proposer une expérience personnalisée.

## Objectifs

- Développer un système de recommandation basé sur l'intelligence artificielle.
- Personnaliser les recommandations pour chaque utilisateur.
- Exploiter les informations textuelles des services.
- Analyser les interactions entre les utilisateurs et les services.
- Combiner plusieurs méthodes de recommandation.
- Permettre une mise à jour des recommandations au fur et à mesure des nouvelles interactions.

## Données

Le projet utilise des données simulées représentant :

- des services ;
- des demandes utilisateurs ;
- des interactions entre utilisateurs et services.

Les données permettent de tester le système dans différentes situations et d'évaluer la pertinence des recommandations.

## Méthodologie

Le système repose sur plusieurs étapes :

1. Génération et préparation des données.
2. Prétraitement des informations textuelles.
3. Représentation des textes avec TF-IDF.
4. Réduction de dimension avec SVD.
5. Génération de recommandations basées sur le contenu.
6. Analyse des interactions utilisateurs-services.
7. Combinaison des différentes approches.
8. Génération des recommandations personnalisées.
9. Mise à jour du système à partir des nouvelles interactions.

## Approche Content-Based

L'approche basée sur le contenu utilise les caractéristiques textuelles des services afin d'identifier les services similaires.

La méthode TF-IDF permet de transformer les descriptions textuelles en représentations numériques.

La similarité entre les services peut ensuite être utilisée pour proposer des recommandations similaires à celles qui correspondent aux intérêts de l'utilisateur.

## Réduction de dimension

La méthode SVD (Singular Value Decomposition) est utilisée afin de réduire la dimension des représentations et de faciliter l'exploitation des données.

Cette étape permet de travailler avec une représentation plus compacte des informations.

## Système hybride

Le système combine plusieurs sources d'information afin d'améliorer la personnalisation des recommandations.

L'approche hybride prend notamment en compte :

- les caractéristiques des services ;
- les interactions des utilisateurs ;
- les préférences observées ;
- les similarités entre services.

Cette combinaison permet de produire des recommandations plus adaptées au profil de chaque utilisateur.

## Analyse NLP

Le traitement du langage naturel est utilisé pour exploiter les informations textuelles associées aux services.

Les techniques NLP permettent notamment de transformer les descriptions textuelles en caractéristiques exploitables par le système de recommandation.

## Apprentissage en ligne

Le système prend en compte les nouvelles interactions afin d'actualiser progressivement les recommandations.

Cette approche permet d'adapter les résultats en fonction de l'évolution des préférences des utilisateurs.

## Mode interactif

Le projet comprend également un mode interactif permettant de tester le système de recommandation et d'obtenir des recommandations personnalisées.

## Technologies utilisées

- Python
- NumPy
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF
- SVD
- Machine Learning
- Recommendation Systems

## Architecture générale

```text
Données utilisateurs et services
            |
            v
     Prétraitement
            |
            v
      Analyse NLP
            |
            v
         TF-IDF
            |
            v
          SVD
            |
            v
  Recommandation Content-Based
            |
            v
 Analyse des interactions utilisateurs
            |
            v
    Système de recommandation
            |
            v
 Recommandations personnalisées
```

## Compétences mobilisées

- Développement Python.
- Machine Learning.
- Natural Language Processing.
- Systèmes de recommandation.
- Traitement et analyse de données.
- Vectorisation TF-IDF.
- Réduction de dimension avec SVD.
- Personnalisation des recommandations.
- Conception d'un système hybride.

## Résultats

Le projet permet de générer des recommandations personnalisées à partir des caractéristiques des services et des interactions utilisateurs.

L'utilisation d'une approche hybride permet de combiner différentes sources d'information afin d'améliorer la pertinence des recommandations.

## Auteur

Khadija Belbaraka

Data & AI | Python | Machine Learning | NLP | Recommendation Systems
