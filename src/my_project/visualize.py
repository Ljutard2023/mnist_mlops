import matplotlib.pyplot as plt
import torch
import typer
from .model import MyAwesomeModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    print(f"Loading model from {model_checkpoint}...")
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    # Astuce : on remplace la dernière couche par "Identité" pour récupérer les features
    # avant la classification finale
    model.fc1 = torch.nn.Identity()

    # Chargement manuel des données de test traitées
    # (Chemin relatif depuis la racine du projet)
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            images = images.to(DEVICE)
            predictions = model(images)
            embeddings.append(predictions.cpu())
            targets.append(target)

        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    print("Running t-SNE (this might take a moment)...")
    if embeddings.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i), alpha=0.6)
    plt.legend()
    plt.title(f"t-SNE of Model Embeddings")

    output_path = f"reports/figures/{figure_name}"
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {output_path}")


if __name__ == "__main__":
    typer.run(visualize)
