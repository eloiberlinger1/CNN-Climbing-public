import os
import sys
from pathlib import Path

# Ajouter le chemin racine du projet au PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np
import json
from datetime import datetime
import torch
from tools.wall_predictor.predict import WallPredictor
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Dimensions cibles pour le modèle
TARGET_WIDTH = 512
TARGET_HEIGHT = 683
MARGIN_SIZE = 50  # Taille de la marge en pixels

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
            if "corners_annotation_date" in self.annotation_state["annotated_images"][image_name]:
                return True
            
        # Double vérification dans le dossier des annotations
        image_annot_dir = self.annotations_dir / image_name / "corners"
        if not image_annot_dir.exists():
            return False
            
        json_exists = (image_annot_dir / "corners.json").exists()
        
        # Si les fichiers existent mais ne sont pas dans le state, mettre à jour le state
        if json_exists:
            if image_name not in self.annotation_state["annotated_images"]:
                self.annotation_state["annotated_images"][image_name] = {}
                
            self.annotation_state["annotated_images"][image_name].update({
                "corners_annotation_date": datetime.now().isoformat(),
                "corners_file": str((image_annot_dir / "corners.json").relative_to(self.dataset_dir))
            })
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
                
            # Déplacer les annotations qui ne sont pas dans annotated_images
            if "annotated_images" not in self.annotation_state:
                self.annotation_state["annotated_images"] = {}
            
            # Déplacer les entrées qui sont à la racine vers annotated_images
            keys_to_move = []
            for key, value in self.annotation_state.items():
                if isinstance(value, dict) and "wall_annotation_date" in value:
                    keys_to_move.append(key)
            
            for key in keys_to_move:
                if key not in self.annotation_state["annotated_images"]:
                    self.annotation_state["annotated_images"][key] = self.annotation_state.pop(key)
            
            # S'assurer que toutes les clés nécessaires existent
            if "total_images" not in self.annotation_state:
                self.annotation_state["total_images"] = 0
            if "annotated_count" not in self.annotation_state:
                self.annotation_state["annotated_count"] = 0
            
            # Compter les images qui ont des coins annotés
            corners_count = sum(
                1 for img in self.annotation_state["annotated_images"].values()
                if any(key in img for key in ["corners_annotation_date", "corner_annotation_date", "corners_lines_file"])
            )
            self.annotation_state["annotated_corners_count"] = corners_count
            
            # Corriger les anciennes entrées qui utilisaient corner_annotation_date
            for img_data in self.annotation_state["annotated_images"].values():
                if "corner_annotation_date" in img_data:
                    img_data["corners_annotation_date"] = img_data.pop("corner_annotation_date")
                if "corner_file" in img_data:
                    img_data["corners_file"] = img_data.pop("corner_file")
        else:
            self.annotation_state = {
                "annotated_images": {},
                "total_images": 0,
                "annotated_count": 0,
                "annotated_corners_count": 0
            }
            
        self.save_annotation_state()
        
        # Vérifier et mettre à jour l'état des annotations
        self._verify_and_update_state()
            
        # Afficher les statistiques
        print(f"\nStatistiques d'annotation des coins:")
        print(f"Dataset: {self.annotation_state['dataset_name']}")
        print(f"Images trouvées: {self.annotation_state['total_images']}")
        print(f"Images avec coins annotés: {self.count_corner_annotations()}")
        print(f"Images restantes: {self.annotation_state['total_images'] - self.count_corner_annotations()}")
    
    def count_corner_annotations(self):
        """Compte le nombre d'images avec des coins annotés."""
        return sum(1 for img in self.annotation_state["annotated_images"].values() 
                  if "corners_annotation_date" in img)
    
    def _verify_and_update_state(self):
        """Vérifie et met à jour l'état des annotations."""
        # Vérifier les fichiers d'annotation existants
        for image_dir in self.annotations_dir.iterdir():
            if not image_dir.is_dir():
                continue
                
            corners_dir = image_dir / "corners"
            if not corners_dir.exists():
                continue
                
            json_file = corners_dir / "corners.json"
            
            if json_file.exists():
                image_name = image_dir.name
                if image_name not in self.annotation_state["annotated_images"]:
                    self.annotation_state["annotated_images"][image_name] = {}
                    
                self.annotation_state["annotated_images"][image_name].update({
                    "corners_annotation_date": datetime.now().isoformat(),
                    "corners_file": str(json_file.relative_to(self.dataset_dir))
                })
        
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
        if image_name not in self.annotation_state["annotated_images"]:
            self.annotation_state["annotated_images"][image_name] = {}
            
        self.annotation_state["annotated_images"][image_name].update({
            "corners_annotation_date": datetime.now().isoformat(),
            "corners_file": str((self.annotations_dir / image_name / "corners" / "corners.json").relative_to(self.dataset_dir))
        })
        
        self.annotation_state["last_updated"] = datetime.now().isoformat()
        self.save_annotation_state()
        
        # Afficher les statistiques mises à jour
        corners_annotated = self.count_corner_annotations()
        remaining = self.annotation_state["total_images"] - corners_annotated
        print(f"\nProgression: {corners_annotated}/{self.annotation_state['total_images']} images avec coins annotés")
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
            print(f"Total d'images avec coins annotés: {self.count_corner_annotations()}")
            return None
        
        next_image = sorted(unannotated)[0]
        print(f"\nChargement de l'image: {next_image.name}")
        return str(next_image)

class CornerAnnotator:
    def __init__(self, image_path, wall_model_path, dataset_manager):
        # Charger l'image originale
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
            
        # Redimensionner l'image aux dimensions cibles
        if self.original_image.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
            self.original_image = cv2.resize(self.original_image, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Créer une image avec marge pour l'affichage
        self.display_width = TARGET_WIDTH + 2 * MARGIN_SIZE
        self.display_height = TARGET_HEIGHT + 2 * MARGIN_SIZE
        self.display_image = np.ones((self.display_height, self.display_width, 3), dtype=np.uint8) * 255
        
        # Placer l'image au centre avec la marge
        self.display_image[MARGIN_SIZE:MARGIN_SIZE+TARGET_HEIGHT, 
                          MARGIN_SIZE:MARGIN_SIZE+TARGET_WIDTH] = self.original_image
        
        self.image_path = Path(image_path)
        self.dataset_manager = dataset_manager
        self.window_name = "Corner Annotator"
        
        # Prédire le masque du mur
        self.wall_predictor = WallPredictor(wall_model_path)
        self.wall_mask = self.wall_predictor.predict(image_path, show_result=False)
        self.wall_polygon = self.wall_predictor.extract_polygons(self.wall_mask)
        
        # État de l'annotation
        self.corners = []  # Liste des angles [(x, y, θ, length), ...]
        self.current_corner = None
        self.drawing_line = False
        self.selected_corner = None
        
        # Configuration
        self.point_color = (0, 255, 0)  # Vert
        self.line_color = (255, 0, 0)   # Bleu
        self.selected_color = (0, 0, 255)  # Rouge
        self.point_size = 5
        self.line_thickness = 2
        
        # Configuration de la fenêtre
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def mouse_callback(self, event, x, y, flags, param):
        # Ajuster les coordonnées en tenant compte de la marge
        x = x - MARGIN_SIZE
        y = y - MARGIN_SIZE
        
        # Vérifier si le clic est dans les limites de l'image
        if not (0 <= x < TARGET_WIDTH and 0 <= y < TARGET_HEIGHT):
            return
            
        # Vérifier si le point est sur le mur
        if not self.is_point_on_wall(x, y):
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing_line:
                # Début d'un nouvel angle
                self.current_corner = (x, y)
                self.drawing_line = True
            else:
                # Fin de la ligne
                end_x, end_y = x, y
                start_x, start_y = self.current_corner
                
                # Calculer l'angle et la longueur
                dx = end_x - start_x
                dy = end_y - start_y
                theta = np.arctan2(dy, dx)
                length = np.sqrt(dx*dx + dy*dy)
                
                # Sauvegarder l'angle
                self.corners.append((start_x, start_y, theta, length))
                self.drawing_line = False
                self.current_corner = None
                self._update_display()
                
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_line:
            self._update_display(temp_end=(x, y))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Sélectionner le coin le plus proche pour suppression
            self.select_nearest_corner(x, y)
            
    def is_point_on_wall(self, x, y):
        """Vérifie si un point est sur le mur."""
        return self.wall_mask[y, x] > 0
        
    def select_nearest_corner(self, x, y):
        """Sélectionne le coin le plus proche du point (x, y)."""
        if not self.corners:
            return
            
        distances = [(i, np.sqrt((c[0]-x)**2 + (c[1]-y)**2)) 
                    for i, c in enumerate(self.corners)]
        nearest_idx, _ = min(distances, key=lambda x: x[1])
        
        # Supprimer le coin sélectionné
        self.corners.pop(nearest_idx)
        self._update_display()
        
    def _update_display(self, temp_end=None):
        """Met à jour l'affichage avec les annotations."""
        display = self.display_image.copy()
        
        # Dessiner le polygone du mur
        if self.wall_polygon:
            # Ajuster les coordonnées pour la marge
            adjusted_polygon = [(x + MARGIN_SIZE, y + MARGIN_SIZE) for x, y in self.wall_polygon]
            pts = np.array(adjusted_polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
        
        # Dessiner tous les angles existants
        for i, (x, y, theta, length) in enumerate(self.corners):
            # Ajuster les coordonnées pour la marge
            adj_x = x + MARGIN_SIZE
            adj_y = y + MARGIN_SIZE
            
            # Point central
            cv2.circle(display, (int(adj_x), int(adj_y)), self.point_size, self.point_color, -1)
            
            # Ligne de direction
            end_x = int(adj_x + length * np.cos(theta))
            end_y = int(adj_y + length * np.sin(theta))
            cv2.line(display, (int(adj_x), int(adj_y)), (end_x, end_y), self.line_color, self.line_thickness)
            
            # Numéro de l'angle
            cv2.putText(display, str(i+1), (int(adj_x)+10, int(adj_y)+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.line_color, 2)
        
        # Dessiner la ligne en cours de création
        if self.drawing_line and self.current_corner and temp_end:
            # Ajuster les coordonnées pour la marge
            adj_start = (self.current_corner[0] + MARGIN_SIZE, self.current_corner[1] + MARGIN_SIZE)
            adj_end = (temp_end[0] + MARGIN_SIZE, temp_end[1] + MARGIN_SIZE)
            
            cv2.circle(display, adj_start, self.point_size, self.point_color, -1)
            cv2.line(display, adj_start, adj_end, self.line_color, self.line_thickness)
        
        # Afficher les instructions
        instructions = [
            "Instructions:",
            "- Clic gauche: Placer/terminer un angle",
            "- Clic droit: Supprimer l'angle le plus proche",
            "- 's': Sauvegarder",
            "- 'r': Réinitialiser",
            "- 'q': Quitter",
            f"Angles annotés: {len(self.corners)}"
        ]
        
        y = MARGIN_SIZE + 30
        for instruction in instructions:
            cv2.putText(display, instruction, (MARGIN_SIZE + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 25
        
        cv2.imshow(self.window_name, display)
    
    def save_annotation(self):
        """Sauvegarde les annotations."""
        if not self.corners:
            print("Aucun angle à sauvegarder")
            return False
            
        # Créer le répertoire de sortie pour cette image
        base_name = self.image_path.stem
        image_annot_dir = self.dataset_manager.annotations_dir / base_name / "corners"
        image_annot_dir.mkdir(parents=True, exist_ok=True)
        
        # Les coordonnées sont déjà dans la bonne échelle
        annotation_data = {
            "image_name": self.image_path.name,
            "corners": [
                {
                    "position": [float(x), float(y)],
                    "angle": float(theta),
                    "length": float(length)
                }
                for x, y, theta, length in self.corners
            ],
            "wall_polygon": self.wall_polygon,
            "image_size": [TARGET_WIDTH, TARGET_HEIGHT],
            "timestamp": datetime.now().isoformat()
        }
        
        # Sauvegarder au format JSON
        json_path = image_annot_dir / "corners.json"
        with open(json_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
            
        # Sauvegarder la visualisation
        vis_path = image_annot_dir / "visualization.png"
        # Sauvegarder l'image sans la marge
        vis_image = self.original_image.copy()
        for i, (x, y, theta, length) in enumerate(self.corners):
            # Point central
            cv2.circle(vis_image, (int(x), int(y)), self.point_size, self.point_color, -1)
            # Ligne de direction
            end_x = int(x + length * np.cos(theta))
            end_y = int(y + length * np.sin(theta))
            cv2.line(vis_image, (int(x), int(y)), (end_x, end_y), self.line_color, self.line_thickness)
            # Numéro de l'angle
            cv2.putText(vis_image, str(i+1), (int(x)+10, int(y)+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.line_color, 2)
        
        cv2.imwrite(str(vis_path), vis_image)
        
        print(f"\nAnnotations sauvegardées:")
        print(f"JSON: {json_path}")
        print(f"Visualisation: {vis_path}")
        return True
    
    def run(self):
        """Lance l'interface d'annotation."""
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\nAnnotation des angles pour: {self.image_path.name}")
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                return False
            elif key == ord('r'):
                self.corners = []
                self.current_corner = None
                self.drawing_line = False
                self._update_display()
            elif key == ord('s'):
                if self.save_annotation():
                    self.dataset_manager.mark_as_annotated(self.image_path.stem)
                    return True
        
        cv2.destroyAllWindows()

class WallAnnotationTool:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = self.dataset_path / "annotations"
        self.current_image = None
        self.current_polygon = None
        self.current_lines = []
        self.drawing = False
        self.start_point = None
        
        # Créer le répertoire d'annotations s'il n'existe pas
        self.annotations_path.mkdir(parents=True, exist_ok=True)
        
        # Charger l'état des annotations
        self.annotation_state_file = self.dataset_path / "annotation_state.json"
        self.load_annotation_state()
        
        # Afficher les instructions
        self.print_instructions()
        
        # Setup GUI
        self.setup_gui()
        
        # Charger la première image non annotée
        self.load_next_image()

    def print_instructions(self):
        print("\nInstructions d'utilisation :")
        print("-----------------------------")
        print("Souris :")
        print("- Clic gauche + maintenir : Dessiner une ligne")
        print("- Relâcher clic gauche : Terminer la ligne")
        print("\nClavier :")
        print("- 's' : Sauvegarder et passer à l'image suivante")
        print("- 'z' : Annuler la dernière ligne")
        print("- 'q' : Quitter le programme")
        print("-----------------------------\n")

    def load_annotation_state(self):
        """Charge l'état des annotations depuis le fichier JSON."""
        if self.annotation_state_file.exists():
            with open(self.annotation_state_file, 'r') as f:
                self.annotation_state = json.load(f)
                
            # Déplacer les annotations qui ne sont pas dans annotated_images
            if "annotated_images" not in self.annotation_state:
                self.annotation_state["annotated_images"] = {}
            
            # Déplacer les entrées qui sont à la racine vers annotated_images
            keys_to_move = []
            for key, value in self.annotation_state.items():
                if isinstance(value, dict) and "wall_annotation_date" in value:
                    keys_to_move.append(key)
            
            for key in keys_to_move:
                if key not in self.annotation_state["annotated_images"]:
                    self.annotation_state["annotated_images"][key] = self.annotation_state.pop(key)
            
            # S'assurer que toutes les clés nécessaires existent
            if "total_images" not in self.annotation_state:
                self.annotation_state["total_images"] = 0
            if "annotated_count" not in self.annotation_state:
                self.annotation_state["annotated_count"] = 0
            
            # Compter les images qui ont des coins annotés
            corners_count = sum(
                1 for img in self.annotation_state["annotated_images"].values()
                if any(key in img for key in ["corners_annotation_date", "corner_annotation_date", "corners_lines_file"])
            )
            self.annotation_state["annotated_corners_count"] = corners_count
            
            # Corriger les anciennes entrées qui utilisaient corner_annotation_date
            for img_data in self.annotation_state["annotated_images"].values():
                if "corner_annotation_date" in img_data:
                    img_data["corners_annotation_date"] = img_data.pop("corner_annotation_date")
                if "corner_file" in img_data:
                    img_data["corners_file"] = img_data.pop("corner_file")
        else:
            self.annotation_state = {
                "annotated_images": {},
                "total_images": 0,
                "annotated_count": 0,
                "annotated_corners_count": 0
            }
            
        self.save_annotation_state()

    def save_annotation_state(self):
        """Sauvegarde l'état des annotations."""
        with open(self.annotation_state_file, 'w') as f:
            json.dump(self.annotation_state, f, indent=2)

    def load_next_image(self):
        """Charge la prochaine image non annotée."""
        # Parcourir les dossiers d'annotation
        for img_dir in self.annotations_path.iterdir():
            if img_dir.is_dir():
                # Vérifier si l'image n'a pas déjà des coins annotés
                if (img_dir.name not in self.annotation_state.get("annotated_images", {}) or
                    "corners_annotation_date" not in self.annotation_state["annotated_images"].get(img_dir.name, {})):
                    # Charger l'image
                    img_path = img_dir / "resized.png"
                    polygon_path = img_dir / "walls" / "polygon.json"
                    
                    if img_path.exists() and polygon_path.exists():
                        # Charger l'image
                        self.current_image = cv2.imread(str(img_path))
                        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                        
                        # Charger le polygone
                        with open(polygon_path, 'r') as f:
                            self.current_polygon = json.load(f)[0]
                        
                        # Créer un masque noir en dehors du polygone
                        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [np.array(self.current_polygon)], 255)
                        self.current_image[mask == 0] = 0
                        
                        # Afficher l'image
                        self.display_image()
                        self.current_dir = img_dir
                        print(f"\nImage chargée : {img_dir.name}")
                        return
        
        print("\nToutes les images ont été annotées!")
        print(f"Total d'images avec coins annotés : {self.annotation_state.get('annotated_corners_count', 0)}")
        self.root.quit()

    def setup_gui(self):
        """Configure l'interface graphique."""
        self.root = tk.Tk()
        self.root.title(f"Outil d'annotation de murs - {self.dataset_path.name}")
        
        # Frame principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas pour l'image avec scrollbars
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Panneau de contrôle
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Boutons
        ttk.Button(self.control_panel, text="Suivant", command=self.save_and_next).pack(pady=5)
        ttk.Button(self.control_panel, text="Annuler dernière ligne", command=self.undo_last_line).pack(pady=5)
        
        # Raccourcis clavier
        self.root.bind('<s>', lambda e: self.save_and_next())
        self.root.bind('<z>', lambda e: self.undo_last_line())
        self.root.bind('<q>', lambda e: self.root.quit())
        
        # Events
        self.canvas.bind("<Button-1>", self.start_line)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_line)
        self.canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        """Gère le redimensionnement de la fenêtre."""
        if self.current_image is not None:
            self.display_image()

    def display_image(self):
        """Affiche l'image et les lignes annotées."""
        if self.current_image is None:
            return
            
        # Convertir l'image pour Tkinter
        image = Image.fromarray(self.current_image)
        self.photo = ImageTk.PhotoImage(image)
        
        # Effacer le canvas
        self.canvas.delete("all")
        
        # Ajuster la taille du canvas à celle de l'image
        self.canvas.config(width=image.width, height=image.height)
        
        # Afficher l'image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Dessiner les lignes existantes
        for line in self.current_lines:
            start_x, start_y = line[0], line[1]
            end_x, end_y = line[2], line[3]
            self.canvas.create_line(start_x, start_y, end_x, end_y,
                                  fill='red', width=2)

    def start_line(self, event):
        """Commence le dessin d'une ligne."""
        self.drawing = True
        self.start_point = (event.x, event.y)

    def draw_line(self, event):
        """Met à jour la ligne en cours de dessin."""
        if not self.drawing:
            return
        
        # Effacer l'ancienne ligne temporaire
        self.canvas.delete("temp_line")
        
        # Dessiner la nouvelle ligne temporaire
        self.canvas.create_line(
            self.start_point[0], self.start_point[1],
            event.x, event.y,
            fill='red', width=2, tags="temp_line"
        )

    def end_line(self, event):
        """Termine le dessin d'une ligne."""
        if not self.drawing:
            return
            
        self.drawing = False
        end_point = (event.x, event.y)
        
        # Ajouter la ligne à la liste
        self.current_lines.append([
            self.start_point[0], self.start_point[1],
            end_point[0], end_point[1]
        ])
        
        # Effacer la ligne temporaire et redessiner toutes les lignes
        self.canvas.delete("temp_line")
        self.display_image()

    def undo_last_line(self):
        """Supprime la dernière ligne dessinée."""
        if self.current_lines:
            self.current_lines.pop()
            self.display_image()

    def save_and_next(self):
        """Sauvegarde les annotations et passe à l'image suivante."""
        if self.current_dir:
            # Créer le répertoire walls s'il n'existe pas
            walls_dir = self.current_dir / "walls"
            walls_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder les annotations
            annotation_file = walls_dir / "lines.json"
            with open(annotation_file, 'w') as f:
                json.dump({
                    "lines": self.current_lines
                }, f)
            
            # Mettre à jour l'état
            if "annotated_images" not in self.annotation_state:
                self.annotation_state["annotated_images"] = {}
            
            # Mettre à jour le compteur uniquement si c'est une nouvelle annotation
            if (self.current_dir.name not in self.annotation_state["annotated_images"] or
                "corners_annotation_date" not in self.annotation_state["annotated_images"][self.current_dir.name]):
                if "annotated_corners_count" not in self.annotation_state:
                    self.annotation_state["annotated_corners_count"] = 0
                self.annotation_state["annotated_corners_count"] += 1
            
            # Mettre à jour les informations d'annotation
            if self.current_dir.name not in self.annotation_state["annotated_images"]:
                self.annotation_state["annotated_images"][self.current_dir.name] = {}
            
            self.annotation_state["annotated_images"][self.current_dir.name].update({
                "corners_annotation_date": datetime.now().isoformat(),
                "corners_lines_file": str(annotation_file.relative_to(self.dataset_path))
            })
            
            self.save_annotation_state()
            
            print(f"\nAnnotations sauvegardées : {annotation_file}")
            print(f"Images avec coins annotés : {self.annotation_state['annotated_corners_count']}")
            
            # Réinitialiser et charger la prochaine image
            self.current_lines = []
            self.load_next_image()

    def run(self):
        """Lance l'application."""
        self.root.mainloop()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Outil d'annotation des droites sur les murs")
    parser.add_argument("dataset", help="Nom du dataset à utiliser")
    args = parser.parse_args()
    
    dataset_path = Path("data") / args.dataset
    if not dataset_path.exists():
        print(f"Erreur : Le dataset '{args.dataset}' n'existe pas dans le dossier data/")
        sys.exit(1)
    
    tool = WallAnnotationTool(dataset_path)
    tool.run()

if __name__ == "__main__":
    main()
