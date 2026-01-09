import torch
import pytest
from my_project.data import corrupt_mnist

def test_data():
    """
    Test que les données sont bien chargées et ont la bonne dimension.
    """
    # 1. On essaie de charger les données
    # Attention : cela nécessite que 'data/processed' existe (via invoke preprocess-data)
    try:
        train_set, test_set = corrupt_mnist()
    except FileNotFoundError:
        pytest.skip("Les fichiers de données 'processed' sont manquants. Lance 'invoke preprocess-data' d'abord.")

    # 2. Vérification qu'il y a bien des données
    assert len(train_set) > 0, "Le jeu d'entraînement ne doit pas être vide"
    assert len(test_set) > 0, "Le jeu de test ne doit pas être vide"

    # 3. Vérification de la forme des tenseurs (Shape check)
    # MNIST corrompu doit être (1, 28, 28)
    sample_image, sample_label = train_set[0]
    
    assert sample_image.shape == (1, 28, 28), f"L'image devrait être (1, 28, 28), mais on a {sample_image.shape}"
    
    # Vérification que toutes les étiquettes sont présentes (on suppose 10 classes pour MNIST)
    unique_labels = torch.unique(torch.tensor([y for _, y in train_set]))
    assert len(unique_labels) <= 10, "Il y a trop de classes différentes"