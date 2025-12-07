import sys
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import json

# Ajouter le dossier parent au path pour pouvoir importer le modèle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.wall_segmentation.wall_model import WallModel

# Dimensions utilisées pour l'entraînement
TARGET_WIDTH = 512
TARGET_HEIGHT = 683

class WallPredictor:
    def __init__(self, checkpoint_path):
        # Charger le modèle et ses poids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WallModel().to(self.device)
        
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Informations sur le modèle
        self.training_info = checkpoint.get('training_info', {})
        print(f"\nModèle chargé: {checkpoint_path}")
        print(f"Entraîné sur: {self.training_info.get('dataset_name', 'N/A')}")
        print(f"Date d'entraînement: {self.training_info.get('training_date', 'N/A')}")
        print(f"Loss finale: {self.training_info.get('final_loss', 'N/A')}")
    
    def extract_polygons(self, mask):
        """Extrait le plus grand polygone à partir du masque binaire."""
        # Trouver les contours dans le masque
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Trouver le plus grand contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplifier le contour en utilisant l'algorithme de Douglas-Peucker
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convertir en liste de points [x, y]
        polygon = [[int(point[0][0]), int(point[0][1])] for point in approx]
        
        # Ne retourner le polygone que s'il a au moins 3 points
        return polygon if len(polygon) >= 3 else None

    def predict(self, image_path, output_dir=None, threshold=0.5, show_result=True):
        """Prédit les murs sur une image."""
        # Charger et préparer l'image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Redimensionner l'image
        image_resized = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        
        # Convertir en tensor
        x = torch.FloatTensor(np.array(image_resized)) / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            pred = self.model(x)
        
        # Convertir la prédiction en masque binaire
        pred_mask = (pred[0, 0].cpu().numpy() > threshold).astype(np.uint8) * 255
        
        # Redimensionner le masque aux dimensions originales si nécessaires
        if original_size != (TARGET_WIDTH, TARGET_HEIGHT):
            pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Sauvegarder les résultats si un dossier de sortie est spécifié
        if output_dir:
            output_dir = Path(output_dir)
            base_name = Path(image_path).stem
            
            # Créer la structure de dossiers pour les prédictions
            image_dir = output_dir / base_name
            walls_dir = image_dir / "walls"
            walls_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder l'image redimensionnée
            resized_path = image_dir / "resized.png"
            image_resized.save(str(resized_path))
            
            # Sauvegarder le masque
            mask_path = walls_dir / "mask.png"
            cv2.imwrite(str(mask_path), pred_mask)
            
            # Extraire et sauvegarder les polygones
            polygon = self.extract_polygons(pred_mask)
            if polygon:
                polygons_path = walls_dir / "polygon.json"
                with open(polygons_path, 'w') as f:
                    json.dump(polygon, f, indent=2)
                print(f"Polygone: {polygons_path}")
            
            # Créer une visualisation avec superposition
            image_np = np.array(image)
            overlay = image_np.copy()
            
            # Dessiner le polygone sur l'overlay
            if polygon:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            
            # Sauvegarder la visualisation
            vis_path = image_dir / "prediction_visualization.png"
            cv2.imwrite(str(vis_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            print(f"\nRésultats sauvegardés dans: {image_dir}")
            print(f"- Image redimensionnée: resized.png")
            print(f"- Masque: walls/mask.png")
            print(f"- Visualisation: prediction_visualization.png")
        
        # Afficher les résultats
        if show_result:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(image)
            plt.title("Image originale")
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(pred_mask, cmap='gray')
            plt.title(f"Prédiction (seuil: {threshold})")
            plt.axis('off')
            
            plt.subplot(133)
            # Dessiner le polygone sur l'image
            overlay = np.array(image).copy()
            if polygon:
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            plt.imshow(overlay)
            plt.title("Polygone détecté")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return pred_mask

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <chemin_modele> <chemin_image> [dossier_sortie]")
        print("Example: python predict.py checkpoints/WallModel_latest.pth images/test.jpg predictions")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        predictor = WallPredictor(model_path)
        predictor.predict(image_path, output_dir=output_dir)
    except Exception as e:
        print(f"Erreur: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 