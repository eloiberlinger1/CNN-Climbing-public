import json
from pathlib import Path
import shutil
from datetime import datetime

def migrate_annotations():
    """
    Migre les anciennes annotations vers la nouvelle structure en utilisant
    le fichier annotations/annotation_state.json existant
    """
    # Chemins des répertoires
    current_dir = Path.cwd()  # tools/wall_annotator
    workspace_dir = current_dir.parent.parent
    
    # Ancien fichier d'état
    old_state_file = workspace_dir / "tools/wall_annotator/annotations" / "annotation_state.json"
    if not old_state_file.exists():
        raise ValueError(f"Fichier d'état non trouvé: {old_state_file}")
        
    print(f"\nMigration des annotations depuis: {old_state_file}")
    
    # Charger l'ancien état (liste de noms d'images)
    with open(old_state_file) as f:
        image_names = json.load(f)
    
    # Nouveau répertoire dataset
    dataset_dir = workspace_dir / "data" / "dataset1"
    annotations_dir = dataset_dir / "annotations"
    
    # Créer le nouveau fichier d'état
    new_state = {
        "dataset_name": "dataset1",
        "creation_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "annotated_images": {},
        "total_images": len(image_names),
        "annotated_count": 0
    }
    
    # Migrer chaque annotation
    print("\nMigration des annotations...")
    for image_name in image_names:
        try:
            print(f"\nTraitement de: {image_name}")
            image_stem = image_name  # Le nom est déjà le stem
            
            # Créer le répertoire pour cette image
            image_annot_dir = annotations_dir / image_stem / "walls"
            image_annot_dir.mkdir(parents=True, exist_ok=True)
            
            # Récupérer les anciens fichiers
            old_annotations_dir = workspace_dir / "tools/wall_annotator/annotations"
            old_polygon_file = old_annotations_dir / f"polygons_{image_stem}.json"
            old_mask_file = old_annotations_dir / f"mask_{image_stem}.png"
            
            if old_polygon_file.exists() and old_mask_file.exists():
                # Copier les fichiers avec les nouveaux noms
                new_polygon_file = image_annot_dir / "polygon.json"
                new_mask_file = image_annot_dir / "mask.png"
                
                shutil.copy2(old_polygon_file, new_polygon_file)
                shutil.copy2(old_mask_file, new_mask_file)
                
                # Mettre à jour l'état
                new_state["annotated_images"][image_name] = {
                    "wall_annotation_date": datetime.now().isoformat(),  # Pas de timestamp dans l'ancien format
                    "wall_polygon_file": str(new_polygon_file.relative_to(dataset_dir))
                }
                new_state["annotated_count"] += 1
                print(f"Migré: {image_name}")
            else:
                print(f"Attention: Fichiers manquants pour {image_name}")
                
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_name}: {str(e)}")
            continue
    
    # Sauvegarder le nouvel état
    new_state_file = dataset_dir / "annotation_state.json"
    with open(new_state_file, 'w') as f:
        json.dump(new_state, f, indent=2)
    
    print(f"\nMigration terminée:")
    print(f"- {new_state['annotated_count']} annotations migrées sur {new_state['total_images']} images")
    print(f"Nouvel état sauvegardé dans: {new_state_file}")

if __name__ == "__main__":
    try:
        migrate_annotations()
    except Exception as e:
        print(f"Erreur lors de la migration: {str(e)}")