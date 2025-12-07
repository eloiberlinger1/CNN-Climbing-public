import cv2
import numpy as np
from pathlib import Path
import argparse

# Dimensions cibles pour le modèle
TARGET_WIDTH = 512
TARGET_HEIGHT = 683

def adapt_image(dataset_name):
    """
    Adapte les images du dataset au format cible.
    - Redimensionne les images à TARGET_WIDTH x TARGET_HEIGHT
    - Convertit toutes les images en PNG
    """
    # Chemins des répertoires
    workspace_dir = Path.cwd().parent.parent
    dataset_dir = workspace_dir / "data" / dataset_name
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset '{dataset_name}' non trouvé dans {workspace_dir}/data/")
    
    print(f"\nAdaptation des images du dataset: {dataset_name}")
    print(f"Dimensions cibles: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    # Parcourir toutes les images
    image_files = list(images_dir.glob("*.[jp][pn][g]"))
    print(f"\nImages trouvées: {len(image_files)}")
    
    for img_path in image_files:
        print(f"\nTraitement de: {img_path.name}")
        
        try:
            # Charger l'image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Erreur: Impossible de charger l'image {img_path}")
                continue
                
            # Obtenir les dimensions originales
            original_height, original_width = image.shape[:2]
            print(f"Dimensions originales: {original_width}x{original_height}")
            
            # Redimensionner l'image
            resized = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
            
            # Créer le répertoire d'annotations si nécessaire
            image_annot_dir = annotations_dir / img_path.stem
            image_annot_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder l'image redimensionnée
            resized_path = image_annot_dir / "resized.png"
            cv2.imwrite(str(resized_path), resized)
            
            # Si l'image originale n'est pas en PNG, la convertir
            if img_path.suffix.lower() != '.png':
                new_path = images_dir / f"{img_path.stem}.png"
                cv2.imwrite(str(new_path), image)
                # Supprimer l'ancienne image
                img_path.unlink()
                print(f"Image convertie en PNG: {new_path.name}")
            
            print(f"Image adaptée sauvegardée: {resized_path}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {img_path}: {str(e)}")
            continue
    
    print("\nAdaptation terminée!")

def main():
    parser = argparse.ArgumentParser(description="Adapte les images au format cible")
    parser.add_argument("dataset_name", help="Nom du dataset (ex: dataset1)")
    args = parser.parse_args()
    
    try:
        adapt_image(args.dataset_name)
    except Exception as e:
        print(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()