# Outil d'Annotation des Murs

Cet outil permet d'annoter facilement les murs dans des images et de générer les masques binaires correspondants pour l'entraînement du modèle de segmentation.

## Prérequis

```bash
pip install opencv-python numpy matplotlib
```

## Structure des fichiers

- `annotator.py` : Outil interactif d'annotation
- `visualizer.py` : Outil de visualisation des annotations
- `annotations/` : Dossier où sont sauvegardées les annotations (créé automatiquement)

## Utilisation

### 1. Annotation des murs

```bash
python annotator.py <chemin_image>
```

#### Contrôles :
- **Clic gauche** : Ajouter un point au polygone
- **Clic droit** : Fermer le polygone (minimum 3 points)
- **'s'** : Sauvegarder l'annotation
- **'r'** : Réinitialiser
- **'q'** : Quitter

### 2. Visualisation des annotations

```bash
python visualizer.py <chemin_image>
```

## Format des annotations

### Masques binaires
- Format : PNG
- Valeurs : 0 (fond) et 255 (mur)
- Taille : Identique à l'image d'entrée

### Données des polygones
- Format : JSON
- Structure :
  ```json
  {
    "polygons": [
      [[x1, y1], [x2, y2], ...],  // Premier polygone
      [[x1, y1], [x2, y2], ...],  // Second polygone
      ...
    ],
    "image_shape": [height, width]
  }
  ```

## Workflow recommandé

1. Préparer les images à annoter dans un dossier
2. Pour chaque image :
   - Lancer l'outil d'annotation
   - Dessiner les polygones autour des murs
   - Sauvegarder les annotations
3. Vérifier les annotations avec l'outil de visualisation
4. Utiliser les masques générés pour l'entraînement du modèle 