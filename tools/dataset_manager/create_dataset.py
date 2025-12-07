import os
from pathlib import Path
import argparse

def create_dataset(dataset_name):
    """
    Crée la structure de base pour un nouveau dataset.
    
    Args:
        dataset_name (str): Nom du dataset à créer
    """
    # Chemin du workspace
    workspace_dir = Path.cwd().parent.parent
    
    # Créer le chemin du dataset
    dataset_dir = workspace_dir / "data" / dataset_name
    images_dir = dataset_dir / "images"
    
    # Créer les répertoires
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset '{dataset_name}' créé avec succès!")
    print(f"\nStructure créée:")
    print(f"{dataset_dir}/")
    print("└── images/")
    print("\nInstructions:")
    print("1. Ajouter les images du dataset dans le répertoire images/")
    print("2. Exécuter tools/image_adapter/adapt.py pour préparer les images")
    print("3. Lancer l'outil d'annotation avec tools/wall_annotator/annotator.py")

def main():
    parser = argparse.ArgumentParser(description="Crée la structure de base pour un nouveau dataset")
    parser.add_argument("dataset_name", help="Nom du dataset à créer (ex: dataset2)")
    args = parser.parse_args()
    
    try:
        create_dataset(args.dataset_name)
    except Exception as e:
        print(f"Erreur lors de la création du dataset: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 