import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Dimensions cibles utilisées pour l'annotation
TARGET_WIDTH = 512
TARGET_HEIGHT = 683

class MaskVisualizer:
    def __init__(self, image_path, annotations_dir="annotations"):
        self.image_path = Path(image_path)
        self.annotations_dir = Path(annotations_dir)
        
        # Charger l'image originale
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Convertir BGR en RGB
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Charger l'image redimensionnée si elle existe
        resized_path = self.annotations_dir / f"image_{self.image_path.stem}.png"
        if resized_path.exists():
            self.resized_image = cv2.imread(str(resized_path))
            self.resized_image = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2RGB)
        else:
            # Redimensionner l'image si nécessaire
            self.resized_image = cv2.resize(self.original_image, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Charger le masque et les annotations
        self.load_annotations()
    
    def load_annotations(self):
        """Charge le masque et les données d'annotation."""
        self.mask = None
        self.polygons = []
        self.normalized_polygons = []
        
        # Charger le masque
        mask_path = self.annotations_dir / f"mask_{self.image_path.stem}.png"
        if mask_path.exists():
            self.mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Charger les polygones
        json_path = self.annotations_dir / f"polygons_{self.image_path.stem}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.polygons = data["polygons"]
                self.normalized_polygons = data.get("normalized_polygons", [])
                self.original_shape = data.get("original_shape")
                self.annotation_date = data.get("annotation_date")
    
    def show(self):
        """Affiche une visualisation complète des annotations."""
        if self.mask is None:
            print("Aucune annotation trouvée pour cette image")
            return
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Image originale avec dimensions originales
        plt.subplot(231)
        plt.imshow(self.original_image)
        plt.title(f"Image originale\n{self.original_image.shape[:2]}")
        plt.axis('off')
        
        # 2. Image redimensionnée
        plt.subplot(232)
        plt.imshow(self.resized_image)
        plt.title(f"Image redimensionnée\n{TARGET_WIDTH}x{TARGET_HEIGHT}")
        plt.axis('off')
        
        # 3. Masque binaire
        plt.subplot(233)
        plt.imshow(self.mask, cmap='gray')
        plt.title("Masque binaire")
        plt.axis('off')
        
        # 4. Superposition du masque sur l'image redimensionnée
        plt.subplot(234)
        overlay = self.resized_image.copy()
        # Créer un masque RGB
        mask_overlay = np.zeros_like(self.resized_image)
        mask_overlay[self.mask > 0] = [0, 255, 0]  # Vert pour les zones masquées
        # Fusionner l'image et le masque
        blended = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
        plt.imshow(blended)
        plt.title("Superposition masque")
        plt.axis('off')
        
        # 5. Image avec polygones
        plt.subplot(235)
        img_with_polygons = self.resized_image.copy()
        for i, polygon in enumerate(self.polygons):
            points = np.array(polygon, dtype=np.int32)
            cv2.polylines(img_with_polygons, [points], True, (255, 0, 0), 2)
            # Afficher le numéro du polygone
            center = np.mean(points, axis=0, dtype=np.int32)
            cv2.putText(img_with_polygons, str(i+1), tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        plt.imshow(img_with_polygons)
        plt.title(f"Polygones ({len(self.polygons)} murs)")
        plt.axis('off')
        
        # 6. Informations
        plt.subplot(236)
        plt.axis('off')
        info_text = [
            f"Nom: {self.image_path.name}",
            f"Dimensions originales: {self.original_image.shape[:2]}",
            f"Dimensions cible: {TARGET_WIDTH}x{TARGET_HEIGHT}",
            f"Nombre de murs: {len(self.polygons)}",
            f"Date d'annotation: {self.annotation_date if hasattr(self, 'annotation_date') else 'N/A'}"
        ]
        plt.text(0.1, 0.5, '\n'.join(info_text), fontsize=10, va='center')
        plt.title("Informations")
        
        plt.tight_layout()
        plt.show()

def visualize_directory(image_path, annotations_dir="annotations"):
    """Visualise les annotations d'une image."""
    try:
        visualizer = MaskVisualizer(image_path, annotations_dir)
        visualizer.show()
    except Exception as e:
        print(f"Erreur lors de la visualisation: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python visualizer.py <chemin_image>")
        sys.exit(1)
    
    visualize_directory(sys.argv[1]) 