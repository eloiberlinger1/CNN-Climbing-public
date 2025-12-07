from PIL import Image
import os

def afficher_dimensions_images(repertoire):
    # Vérifie si le répertoire existe
    if not os.path.exists(repertoire):
        print(f"Le répertoire '{repertoire}' n'existe pas.")
        return

    # Liste tous les fichiers dans le répertoire
    fichiers = os.listdir(repertoire)

    # Filtre les fichiers pour ne garder que les images
    extensions_images = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = [f for f in fichiers if f.lower().endswith(extensions_images)]

    # Affiche les dimensions de chaque image
    for image in images:
        chemin_image = os.path.join(repertoire, image)
        with Image.open(chemin_image) as img:
            largeur, hauteur = img.size
            print(f"Image: {image}, Dimensions: {largeur}x{hauteur}")

# Appel de la fonction avec le répertoire "images"
afficher_dimensions_images("images")
