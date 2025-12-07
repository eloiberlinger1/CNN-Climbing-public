from pathlib import Path
import json
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime

def analyze_dataset(dataset_name="dataset1"):
    """
    Analyse un dataset et affiche des statistiques détaillées.
    """
    # Chemins des répertoires
    workspace_dir = Path.cwd().parent.parent
    dataset_dir = workspace_dir / "data" / dataset_name
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    
    if not dataset_dir.exists():
        print(f"Erreur: Le dataset '{dataset_name}' n'existe pas dans {workspace_dir}/data/")
        return
    
    print(f"\n=== Analyse du dataset: {dataset_name} ===\n")
    
    # 1. Statistiques générales
    print("1. Informations générales:")
    print(f"Chemin du dataset: {dataset_dir}")
    
    # Charger l'état des annotations
    state_file = dataset_dir / "annotation_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"Date de création: {state['creation_date']}")
        print(f"Dernière mise à jour: {state['last_updated']}")
    
    # 2. Statistiques des images
    print("\n2. Statistiques des images:")
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = []
    image_sizes = defaultdict(int)
    
    for ext in image_extensions:
        images.extend(list(images_dir.glob(f"*{ext}")))
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            size = f"{img.shape[1]}x{img.shape[0]}"
            image_sizes[size] += 1
    
    print(f"Nombre total d'images: {len(images)}")
    print("\nRésolutions d'images:")
    for size, count in image_sizes.items():
        print(f"- {size}: {count} images")
    
    # 3. Statistiques des annotations
    print("\n3. Statistiques des annotations:")
    
    # Compter les annotations de murs
    wall_annotations = 0
    total_walls = 0
    wall_sizes = []
    
    for image_dir in annotations_dir.iterdir():
        if not image_dir.is_dir():
            continue
        
        walls_dir = image_dir / "walls"
        if walls_dir.exists():
            wall_annotations += 1
            
            # Analyser les polygones
            polygon_file = walls_dir / "polygon.json"
            if polygon_file.exists():
                with open(polygon_file) as f:
                    data = json.load(f)
                    if "polygons" in data:
                        total_walls += len(data["polygons"])
                        wall_sizes.extend([len(poly) for poly in data["polygons"]])
    
    print(f"Images avec annotations de murs: {wall_annotations}")
    if wall_annotations > 0:
        print(f"Nombre total de murs annotés: {total_walls}")
        print(f"Moyenne de murs par image: {total_walls/wall_annotations:.1f}")
        if wall_sizes:
            print(f"Points moyens par mur: {sum(wall_sizes)/len(wall_sizes):.1f}")
    
    # 4. Progression
    print("\n4. Progression de l'annotation:")
    if len(images) > 0:
        progress = (wall_annotations / len(images)) * 100
        print(f"Progression: {wall_annotations}/{len(images)} images ({progress:.1f}%)")
        print(f"Images restantes: {len(images) - wall_annotations}")
    
    # 5. Structure du dataset
    print("\n5. Structure du dataset:")
    print("data/")
    print(f"└── {dataset_name}/")
    print("    ├── images/")
    print(f"    │   └── {len(images)} images")
    print("    ├── annotations/")
    print(f"    │   └── {wall_annotations} dossiers d'annotations")
    print("    └── annotation_state.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyse un dataset et affiche des statistiques")
    parser.add_argument("--dataset", default="dataset1", help="Nom du dataset à analyser")
    args = parser.parse_args()
    
    analyze_dataset(args.dataset)
