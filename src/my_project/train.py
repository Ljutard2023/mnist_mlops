import matplotlib.pyplot as plt
import torch
import typer

# Note le point devant data et model pour l'import relatif
from .data import corrupt_mnist
from .model import MyAwesomeModel

app = typer.Typer(rich_markup_mode=None, context_settings={"help_option_names": ["-h", "--help"]})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@app.command()
def train(
    lr: float = 1e-3, 
    batch_size: int = 32, 
    epochs: int = 5
):
    """Entraîne le modèle avec les hyperparamètres donnés."""
    print(f"🚀 Training with: lr={lr}, batch_size={batch_size}, epochs={epochs}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    # On sauvegarde dans le dossier models/ à la racine
    torch.save(model.state_dict(), "models/model.pth")

    # On sauvegarde les figures
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    
    print("Training complete!")

if __name__ == "__main__":
    app()
