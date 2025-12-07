import sys
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ajouter le dossier parent au path pour pouvoir importer le modèle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.wall_segmentation.wall_model import WallModel

# Dimensions cibles pour le modèle
TARGET_WIDTH = 512
TARGET_HEIGHT = 683

def test_model_architecture():
    """Test la structure du modèle"""
    print("\n1. Test de la structure du modèle")
    model = WallModel()
    print("Modèle créé avec succès")
    
    # Afficher l'architecture
    print("\nArchitecture du modèle:")
    print(model)
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNombre total de paramètres: {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")
    
    return model

def test_forward_pass(model):
    """Test le forward pass avec différentes tailles d'images"""
    print("\n2. Test du forward pass")
    
    # Test avec les dimensions cibles et des variations
    test_sizes = [
        (1, 3, TARGET_HEIGHT, TARGET_WIDTH),  # Une seule image
        (2, 3, TARGET_HEIGHT, TARGET_WIDTH),  # Batch de 2 images
        (4, 3, TARGET_HEIGHT, TARGET_WIDTH)   # Batch de 4 images
    ]
    
    for size in test_sizes:
        print(f"\nTest avec input de taille {size}")
        x = torch.randn(size)
        try:
            with torch.no_grad():
                out = model(x)
            print(f"Succès! Taille de sortie: {out.shape}")
            print(f"Valeurs min/max: {out.min():.3f}/{out.max():.3f}")
            
            # Vérifier que la sortie a les bonnes dimensions
            expected_shape = (size[0], 1, TARGET_HEIGHT, TARGET_WIDTH)
            assert out.shape == expected_shape, f"Dimensions incorrectes: {out.shape} vs {expected_shape}"
            
        except Exception as e:
            print(f"Erreur: {str(e)}")

def test_skip_connections(model):
    """Test les skip connections"""
    print("\n3. Test des skip connections")
    
    x = torch.randn(1, 3, TARGET_HEIGHT, TARGET_WIDTH)
    
    # Hooks pour récupérer les activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook
    
    # Enregistrer les hooks
    model.enc1.register_forward_hook(get_activation('enc1'))
    model.enc2.register_forward_hook(get_activation('enc2'))
    model.enc3.register_forward_hook(get_activation('enc3'))
    model.enc4.register_forward_hook(get_activation('enc4'))
    
    # Forward pass
    with torch.no_grad():
        out = model(x)
    
    # Vérifier les tailles des activations
    print("\nTailles des features maps:")
    for name, activation in activations.items():
        print(f"{name}: {activation.shape}")
        
    # Vérifier les dimensions des feature maps
    assert activations['enc1'].shape[2:] == (TARGET_HEIGHT, TARGET_WIDTH)
    assert activations['enc2'].shape[2:] == (TARGET_HEIGHT//2, TARGET_WIDTH//2)
    assert activations['enc3'].shape[2:] == (TARGET_HEIGHT//4, TARGET_WIDTH//4)
    assert activations['enc4'].shape[2:] == (TARGET_HEIGHT//8, TARGET_WIDTH//8)

def test_with_sample_image(model, test_image_path=None):
    """Test avec une image réelle"""
    print("\n4. Test avec une image réelle")
    
    if test_image_path is None:
        print("Aucune image de test spécifiée. Création d'une image synthétique...")
        # Créer une image de test aux dimensions cibles
        test_img = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), 'black')
        
        # Dessiner plusieurs rectangles blancs pour simuler des murs
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([(50, 50), (150, 600)], fill='white')
        draw.rectangle([(200, 200), (450, 300)], fill='white')
    else:
        print(f"Utilisation de l'image de test: {test_image_path}")
        # Charger et redimensionner l'image de test
        test_img = Image.open(test_image_path).convert('RGB')
        test_img = test_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convertir en tensor
    x = torch.FloatTensor(np.array(test_img)).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Prédiction
    with torch.no_grad():
        out = model(x)
    
    # Visualisation
    plt.figure(figsize=(15, 10))
    
    plt.subplot(131)
    plt.imshow(test_img)
    plt.title("Image d'entrée")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(out[0, 0].numpy(), cmap='gray')
    plt.title('Prédiction (raw)')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(out[0, 0].numpy() > 0.5, cmap='gray')
    plt.title('Prédiction (seuil 0.5)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_gradient_flow(model):
    """Test le flux des gradients"""
    print("\n5. Test du flux des gradients")
    
    x = torch.randn(1, 3, TARGET_HEIGHT, TARGET_WIDTH)
    target = torch.rand(1, 1, TARGET_HEIGHT, TARGET_WIDTH)
    
    # Forward pass
    out = model(x)
    loss = torch.nn.BCELoss()(out, target)
    
    # Backward pass
    loss.backward()
    
    # Vérifier les gradients
    print("\nGradients des couches:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_norm = param.grad.norm().item() if param.grad is not None else 0
            print(f"{name}: {grad_norm:.5f}")

def main():
    # Récupérer le chemin de l'image de test depuis les arguments
    test_image_path = None
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        if not os.path.exists(test_image_path):
            print(f"ATTENTION: L'image spécifiée n'existe pas: {test_image_path}")
            print("Une image synthétique sera utilisée à la place.")
            test_image_path = None
    
    print(f"=== Tests du WallModel (dimensions: {TARGET_WIDTH}x{TARGET_HEIGHT}) ===")
    
    # 1. Test de l'architecture
    model = test_model_architecture()
    
    # 2. Test du forward pass
    test_forward_pass(model)
    
    # 3. Test des skip connections
    test_skip_connections(model)
    
    # 4. Test avec une image
    test_with_sample_image(model, test_image_path)
    
    # 5. Test des gradients
    test_gradient_flow(model)
    
    print("\nTous les tests terminés!")

if __name__ == "__main__":
    main()
