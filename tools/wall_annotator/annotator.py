import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

# Dimensions cibles pour le modèle
TARGET_WIDTH = 512
TARGET_HEIGHT = 683

class DatasetManager:
    def __init__(self, dataset_name="dataset1"):
        # Chemins des répertoires
        workspace_dir = Path.cwd().parent.parent
        self.dataset_dir = workspace_dir / "data" / dataset_name
        self.images_dir = self.dataset_dir / "images"
        self.annotations_dir = self.dataset_dir / "annotations"
        
        if not self.images_dir.exists():
            raise ValueError(f"Le répertoire d'images '{self.images_dir}' n'existe pas")
        
        # Créer le répertoire d'annotations s'il n'existe pas
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger l'état des annotations au démarrage
        self.load_annotation_state()
    
    def is_image_annotated(self, image_name):
        """Vérifie si une image est déjà annotée."""
        # Vérifier dans le state
        if image_name in self.annotation_state["annotated_images"]:
            return True
            
        # Double vérification dans le dossier des annotations
        image_annot_dir = self.annotations_dir / image_name / "walls"
        if not image_annot_dir.exists():
            return False
            
        mask_exists = (image_annot_dir / "mask.png").exists()
        json_exists = (image_annot_dir / "polygon.json").exists()
        
        # Si les fichiers existent mais ne sont pas dans le state, mettre à jour le state
        if mask_exists and json_exists:
            self.annotation_state["annotated_images"][image_name] = {
                "wall_annotation_date": datetime.now().isoformat(),
                "wall_polygon_file": str((image_annot_dir / "polygon.json").relative_to(self.dataset_dir))
            }
            self.annotation_state["annotated_count"] += 1
            self.save_annotation_state()
            return True
            
        return False
        
    def load_annotation_state(self):
        """Charge l'état des annotations depuis le fichier JSON."""
        self.annotation_state_file = self.dataset_dir / "annotation_state.json"
        
        if self.annotation_state_file.exists():
            with open(self.annotation_state_file, 'r') as f:
                self.annotation_state = json.load(f)
        else:
            # Créer un nouveau fichier d'état
            self.annotation_state = {
                "dataset_name": self.dataset_dir.name,
                "creation_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "annotated_images": {},
                "total_images": len(list(self.images_dir.glob("*.[jp][pn][g]"))),
                "annotated_count": 0
            }
            self.save_annotation_state()
        
        # Vérifier et mettre à jour l'état des annotations
        self._verify_and_update_state()
            
        # Afficher les statistiques
        print(f"\nStatistiques d'annotation:")
        print(f"Dataset: {self.annotation_state['dataset_name']}")
        print(f"Images trouvées: {self.annotation_state['total_images']}")
        print(f"Images annotées: {self.annotation_state['annotated_count']}")
        print(f"Images restantes: {self.annotation_state['total_images'] - self.annotation_state['annotated_count']}")
    
    def _verify_and_update_state(self):
        """Vérifie et met à jour l'état des annotations."""
        # Réinitialiser le compteur
        self.annotation_state["annotated_count"] = 0
        
        # Vérifier les fichiers d'annotation existants
        for image_dir in self.annotations_dir.iterdir():
            if not image_dir.is_dir():
                continue
                
            walls_dir = image_dir / "walls"
            if not walls_dir.exists():
                continue
                
            mask_file = walls_dir / "mask.png"
            json_file = walls_dir / "polygon.json"
            
            if mask_file.exists() and json_file.exists():
                image_name = image_dir.name
                if image_name not in self.annotation_state["annotated_images"]:
                    self.annotation_state["annotated_images"][image_name] = {
                        "wall_annotation_date": datetime.now().isoformat(),
                        "wall_polygon_file": str(json_file.relative_to(self.dataset_dir))
                    }
                self.annotation_state["annotated_count"] += 1
        
        # Mettre à jour le nombre total d'images
        self.annotation_state["total_images"] = len(list(self.images_dir.glob("*.[jp][pn][g]")))
        self.annotation_state["last_updated"] = datetime.now().isoformat()
        
        # Sauvegarder l'état mis à jour
        self.save_annotation_state()
    
    def save_annotation_state(self):
        """Sauvegarde l'état des annotations."""
        with open(self.annotation_state_file, 'w') as f:
            json.dump(self.annotation_state, f, indent=2)
    
    def mark_as_annotated(self, image_name):
        """Marque une image comme annotée."""
        self.annotation_state["annotated_images"][image_name] = {
            "wall_annotation_date": datetime.now().isoformat(),
            "wall_polygon_file": str((self.annotations_dir / image_name / "walls" / "polygon.json").relative_to(self.dataset_dir))
        }
        self.annotation_state["annotated_count"] += 1
        self.annotation_state["last_updated"] = datetime.now().isoformat()
        self.save_annotation_state()
        
        # Afficher les statistiques mises à jour
        remaining = self.annotation_state["total_images"] - self.annotation_state["annotated_count"]
        print(f"\nProgression: {self.annotation_state['annotated_count']}/{self.annotation_state['total_images']} images annotées")
        print(f"Reste à faire: {remaining} images")
    
    def get_next_image(self):
        """Retourne le chemin de la prochaine image non annotée."""
        # Récupérer toutes les images disponibles
        available_images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            available_images.extend(self.images_dir.glob(f"*{ext}"))
        
        if not available_images:
            print("Aucune image trouvée dans le répertoire images/")
            return None
        
        # Filtrer les images non annotées
        unannotated = [img for img in available_images if not self.is_image_annotated(img.stem)]
        
        if not unannotated:
            print("\nToutes les images ont été annotées!")
            print(f"Total d'images annotées: {self.annotation_state['annotated_count']}")
            return None
        
        next_image = sorted(unannotated)[0]
        print(f"\nChargement de l'image: {next_image.name}")
        return str(next_image)

    def delete_image(self, image_name):
        """Supprime une image et ses annotations du dataset."""
        # Chemins des fichiers à supprimer
        image_file = self.images_dir / f"{image_name}.jpg"  # Vérifier aussi .png et .jpeg si nécessaire
        if not image_file.exists():
            image_file = self.images_dir / f"{image_name}.png"
        if not image_file.exists():
            image_file = self.images_dir / f"{image_name}.jpeg"
        
        annot_dir = self.annotations_dir / image_name
        
        # Supprimer l'image originale
        if image_file.exists():
            image_file.unlink()
            print(f"Image supprimée: {image_file}")
        
        # Supprimer le répertoire d'annotations
        if annot_dir.exists():
            import shutil
            shutil.rmtree(annot_dir)
            print(f"Annotations supprimées: {annot_dir}")
        
        # Mettre à jour annotation_state.json
        if image_name in self.annotation_state["annotated_images"]:
            del self.annotation_state["annotated_images"][image_name]
            self.annotation_state["annotated_count"] -= 1
            self.annotation_state["total_images"] -= 1
            self.annotation_state["last_updated"] = datetime.now().isoformat()
            self.save_annotation_state()
            print(f"État des annotations mis à jour")
        
        print(f"\nImage {image_name} supprimée du dataset")
        print(f"Images restantes: {self.annotation_state['total_images']}")
        print(f"Images annotées: {self.annotation_state['annotated_count']}")

class WallAnnotator:
    def __init__(self, image_path, dataset_manager):
        # Charger l'image originale
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Redimensionner l'image aux dimensions cibles
        self.image = cv2.resize(self.original_image, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Normaliser l'image (0-1)
        self.image = self.image.astype(np.float32) / 255.0
        
        self.image_path = Path(image_path)
        self.dataset_manager = dataset_manager
        self.original = self.image.copy()
        self.current_polygon = []
        self.polygons = []
        self.window_name = "Wall Annotator"
        self.mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        
        # Configuration
        self.point_color = (0, 1, 0)  # Vert (normalisé)
        self.line_color = (1, 0, 0)   # Bleu (normalisé)
        self.point_size = 3
        self.line_thickness = 2
        
        # Configuration de la fenêtre en plein écran
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ajouter un point au polygone courant
            self.current_polygon.append((x, y))
            self._update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.current_polygon) > 2:
            # Fermer le polygone courant
            self.polygons.append(self.current_polygon.copy())
            # Mettre à jour le masque
            self._update_mask()
            self.current_polygon = []
            self._update_display()
    
    def _update_mask(self):
        # Créer un masque pour le dernier polygone ajouté
        polygon = np.array(self.polygons[-1], dtype=np.int32)
        cv2.fillPoly(self.mask, [polygon], 255)
    
    def _update_display(self):
        # Créer une copie de l'image pour l'affichage
        display = self.original.copy()
        
        # Convertir en uint8 pour l'affichage
        display_img = (display * 255).astype(np.uint8)
        
        # Afficher le nom de l'image et le nombre de polygones
        cv2.putText(display_img, f"{self.image_path.name} - {len(self.polygons)} murs", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Afficher tous les polygones validés
        for i, polygon in enumerate(self.polygons):
            points = np.array(polygon, dtype=np.int32)
            cv2.polylines(display_img, [points], True, (255, 0, 0), self.line_thickness)
            # Afficher le numéro du polygone
            center = np.mean(points, axis=0, dtype=np.int32)
            cv2.putText(display_img, str(i+1), tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            for point in polygon:
                cv2.circle(display_img, point, self.point_size, (0, 255, 0), -1)
        
        # Afficher le polygone en cours
        if len(self.current_polygon) > 0:
            for point in self.current_polygon:
                cv2.circle(display_img, point, self.point_size, (0, 255, 0), -1)
            
            if len(self.current_polygon) > 1:
                points = np.array(self.current_polygon, dtype=np.int32)
                cv2.polylines(display_img, [points], False, (255, 0, 0), self.line_thickness)
        
        # Afficher l'image et le masque côte à côte
        mask_display = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        combined_display = np.hstack((display_img, mask_display))
        cv2.imshow(self.window_name, combined_display)
    
    def run(self):
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\nAnnotation de: {self.image_path.name}")
        print("\nInstructions:")
        print("- Clic gauche: Ajouter un point au polygone")
        print("- Clic droit: Fermer le polygone (minimum 3 points)")
        print("- 's': Sauvegarder l'annotation et passer à l'image suivante")
        print("- 'r': Réinitialiser")
        print("- 'd': Supprimer l'image du dataset")
        print("- 'q': Quitter")
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                return False
            elif key == ord('r'):
                self.current_polygon = []
                self.polygons = []
                self.mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
                self._update_display()
            elif key == ord('s'):
                if self.polygons:
                    self.save_annotation()
                    self.dataset_manager.mark_as_annotated(self.image_path.stem)
                    return True
                else:
                    print("Aucun polygone à sauvegarder")
            elif key == ord('d'):
                # Demander confirmation
                print("\nÊtes-vous sûr de vouloir supprimer cette image ? (o/n)")
                while True:
                    confirm = cv2.waitKey(0) & 0xFF
                    if confirm == ord('o'):
                        self.dataset_manager.delete_image(self.image_path.stem)
                        return True
                    elif confirm == ord('n'):
                        print("Suppression annulée")
                        break
    
    def save_annotation(self):
        if not self.polygons:
            return
        
        # Créer le répertoire de sortie pour cette image
        base_name = self.image_path.stem
        image_annot_dir = self.dataset_manager.annotations_dir / base_name / "walls"
        image_annot_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le masque
        mask_path = image_annot_dir / "mask.png"
        cv2.imwrite(str(mask_path), self.mask)
        
        # Sauvegarder uniquement les polygones
        polygons_path = image_annot_dir / "polygon.json"
        with open(polygons_path, 'w') as f:
            json.dump(self.polygons, f, indent=2)
        
        print(f"\nAnnotations sauvegardées:")
        print(f"- Masque: {mask_path}")
        print(f"- Polygones: {polygons_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Outil d'annotation de murs")
    parser.add_argument("--dataset", default="dataset1", help="Nom du dataset à annoter")
    args = parser.parse_args()
    
    try:
        # Initialiser le gestionnaire de dataset
        dataset_manager = DatasetManager(dataset_name=args.dataset)
        
        while True:
            # Obtenir la prochaine image à annoter
            image_path = dataset_manager.get_next_image()
            if image_path is None:
                break
            
            try:
                # Créer et exécuter l'annotateur
                annotator = WallAnnotator(image_path, dataset_manager)
                if not annotator.run():
                    break
            except ValueError as e:
                # Erreur de chargement d'image
                print(f"\nErreur: {str(e)}")
                print("\nVoulez-vous supprimer cette image du dataset ? (o/n)")
                while True:
                    response = input().lower()
                    if response == 'o':
                        dataset_manager.delete_image(Path(image_path).stem)
                        print("Image supprimée du dataset.")
                        break
                    elif response == 'n':
                        print("Image conservée.")
                        break
                    else:
                        print("Veuillez répondre par 'o' ou 'n'")
                continue
        
        cv2.destroyAllWindows()
        print("\nAnnotation terminée!")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 