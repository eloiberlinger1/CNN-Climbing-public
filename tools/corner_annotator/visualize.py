import os
import sys
from pathlib import Path

# Ajouter le chemin racine du projet au PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np
import json
import argparse

def visualize_annotations(dataset_name, image_name):
    # Chemins des répertoires
    dataset_path = Path("data") / dataset_name
    annotations_path = dataset_path / "annotations" / image_name
    
    # Vérifier que l'image existe
    image_path = annotations_path / "resized.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image non trouvée : {image_path}")
    
    # Vérifier que les annotations existent
    lines_path = annotations_path / "walls" / "lines.json"
    if not lines_path.exists():
        raise FileNotFoundError(f"Annotations non trouvées : {lines_path}")
    
    # Charger l'image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
    # Charger les annotations
    with open(lines_path, 'r') as f:
        annotations = json.load(f)
    
    # Dessiner les lignes
    for i, line in enumerate(annotations["lines"]):
        # Les coordonnées sont stockées comme [start_x, start_y, end_x, end_y]
        start_point = (int(line[0]), int(line[1]))
        end_point = (int(line[2]), int(line[3]))
        
        # Dessiner la ligne
        cv2.line(image, start_point, end_point, (0, 0, 255), 2)
        
        # Ajouter un numéro à côté de chaque ligne
        mid_point = ((start_point[0] + end_point[0]) // 2,
                    (start_point[1] + end_point[1]) // 2)
        cv2.putText(image, str(i+1), mid_point,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Afficher l'image
    window_name = f"Annotations - {image_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    
    print("\nInstructions :")
    print("- Appuyez sur 'q' pour quitter")
    print("- Appuyez sur 's' pour sauvegarder la visualisation")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Sauvegarder la visualisation
            output_path = annotations_path / "walls" / "visualization.png"
            cv2.imwrite(str(output_path), image)
            print(f"\nVisualisation sauvegardée : {output_path}")
    
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualisation des annotations de droites")
    parser.add_argument("dataset", help="Nom du dataset")
    parser.add_argument("image", help="Nom de l'image à visualiser")
    args = parser.parse_args()
    
    try:
        visualize_annotations(args.dataset, args.image)
    except Exception as e:
        print(f"Erreur : {str(e)}")

if __name__ == "__main__":
    main() 