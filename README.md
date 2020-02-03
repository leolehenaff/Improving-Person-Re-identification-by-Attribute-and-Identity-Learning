# Improving-Person-Re-identification-by-Attribute-and-Identity-Learning
Deep-Learning (CNN) project

Nous avons implémenté:

- Un modèle pour la classfications d'attributs (~90% d'accuracy) après une quinzaine d'epochs.
- L'augmentation d'images (cropping, distortion, noise etc).
- Un modèle pour la classification d'ID combiné avec la classification d'attributs. Le modèle s'entraîne avec 2 losses en même temps et obtient ~94% d'accuracy en validation sur la prédiction d'ID après environ 40 epochs, après un entraînement sur 75% des données.
- Le module d'attribute re-weighting, qui est en fait un simple layer Dense dont la sortie est multipliée par les attributs
- Le calcul de la distance euclidienne, cependant celui-ci donne de mauvais résultats: peut-être que nous sélectionnons le mauvais vecteur de features. Il est compliqué de calculer l'accuracy réelle car comparer chaque image à 25 000 autres prend trop de temps et de puissance de calculs.


![Schema of the Project](https://github.com/leolehenaff/Improving-Person-Re-identification-by-Attribute-and-Identity-Learning/blob/master/schema.jpg)
