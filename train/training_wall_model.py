import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from datetime import datetime
import json
from tqdm import tqdm

# Ajouter le dossier parent au path pour pouvoir importer le modèle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.wall_segmentation.wall_model import WallModel

# Dimensions cibles pour le modèle
TARGET_WIDTH = 512
TARGET_HEIGHT = 683

class WallDataset(Dataset):
    def __init__(self, dataset_name):
        # Chemins des répertoires
        workspace_dir = Path.cwd().parent.parent
        self.dataset_dir = workspace_dir / "data" / dataset_name
        self.annotations_dir = self.dataset_dir / "annotations"
        
        if not self.dataset_dir.exists():
            raise ValueError(f"Le dataset {dataset_name} n'existe pas dans {workspace_dir}/data/")
        
        # Charger l'état des annotations
        annotation_state_file = self.dataset_dir / "annotation_state.json"
        if not annotation_state_file.exists():
            raise ValueError(f"Fichier d'état des annotations non trouvé: {annotation_state_file}")
            
        with open(annotation_state_file, 'r') as f:
            annotation_state = json.load(f)
        
        # Initialiser les listes pour les chemins des fichiers
        self.resized_images = []
        self.mask_files = []
        
        print("\nChargement du dataset...")
        total_annotated = len(annotation_state["annotated_images"])
        loaded_count = 0
        
        # Parcourir les images annotées
        for image_name in annotation_state["annotated_images"]:
            # Vérifier les fichiers nécessaires pour l'entraînement
            image_dir = self.annotations_dir / image_name
            walls_dir = image_dir / "walls"
            
            resized_file = image_dir / "resized.png"  # resized.png dans le répertoire principal
            mask_file = walls_dir / "mask.png"        # mask.png dans le sous-répertoire walls
            
            if resized_file.exists() and mask_file.exists():
                self.resized_images.append(resized_file)
                self.mask_files.append(mask_file)
                loaded_count += 1
            else:
                missing = []
                if not resized_file.exists():
                    missing.append("resized.png")
                if not mask_file.exists():
                    missing.append("mask.png")
                print(f"Image ignorée {image_name}: fichiers manquants ({', '.join(missing)})")
        
        if not self.resized_images:
            print("\nAucune image valide trouvée. Structure attendue:")
            print(f"{self.dataset_dir}/")
            print("└── annotations/")
            print("    └── <nom_image>/")
            print("        ├── resized.png     (image redimensionnée)")
            print("        └── walls/")
            print("            ├── mask.png    (masque d'annotation)")
            print("            └── polygon.json (polygones annotés)")
            raise ValueError("Aucune image valide pour l'entraînement")
        
        print(f"\nStatistiques du dataset:")
        print(f"- Images annotées: {total_annotated}")
        print(f"- Images valides pour l'entraînement: {loaded_count}")
        print(f"- Images ignorées: {total_annotated - loaded_count}")
        
        # Transformations (pas besoin de resize car on utilise resized.png)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.resized_images)
    
    def __getitem__(self, idx):
        # Charger l'image redimensionnée et le masque
        image = Image.open(self.resized_images[idx]).convert('RGB')
        mask = Image.open(self.mask_files[idx]).convert('L')
        
        # Appliquer les transformations
        image = self.transform(image)
        mask = torch.FloatTensor(np.array(mask)) / 255.0
        mask = mask.unsqueeze(0)  # Ajouter la dimension des channels
        
        return image, mask

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()
    
    # Pour suivre la progression
    best_loss = float('inf')
    history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Transférer les données sur le device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Mettre à jour les statistiques
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculer la perte moyenne de l'epoch
        epoch_loss = running_loss / len(train_loader)
        history.append(epoch_loss)
        
        print(f'\nEpoch {epoch+1} - Loss: {epoch_loss:.4f}')
        
        # Sauvegarder le meilleur modèle
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
    
    return best_state, history

def main():
    # Vérifier les arguments
    if len(sys.argv) != 2:
        print("Usage: python training_wall_model.py <nom_dataset>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Paramètres d'entraînement
    #mps = apple silicon
    #cuda = GPU NVIDIA
    #cpu = CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Utilisation de: {device}")
    
    # Créer le modèle
    model = WallModel().to(device)
    print("Modèle créé")
    
    # Charger le dataset
    try:
        dataset = WallDataset(dataset_name)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        print("Dataset chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {str(e)}")
        sys.exit(1)
    
    # Définir le critère et l'optimiseur
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entraîner le modèle
    print("\nDémarrage de l'entraînement...")
    best_state, history = train_model(model, train_loader, criterion, optimizer, device)
    
    # Créer le dossier de checkpoint s'il n'existe pas
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Sauvegarder le modèle avec timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    checkpoint_path = checkpoint_dir / f"WallModel_{timestamp}.pth"
    
    # Ajouter des informations supplémentaires au checkpoint
    best_state['training_info'] = {
        'dataset_name': dataset_name,
        'training_date': timestamp,
        'final_loss': best_state['loss'],
        'image_size': [TARGET_HEIGHT, TARGET_WIDTH],
        'device_used': str(device)
    }
    
    # Sauvegarder le checkpoint
    torch.save(best_state, checkpoint_path)
    print(f"\nModèle sauvegardé: {checkpoint_path}")
    
    # Sauvegarder l'historique des pertes
    history_path = checkpoint_dir / f"WallModel_{timestamp}_history.json"
    with open(history_path, 'w') as f:
        json.dump({'loss_history': history}, f)
    print(f"Historique d'entraînement sauvegardé: {history_path}")

if __name__ == "__main__":
    main()
