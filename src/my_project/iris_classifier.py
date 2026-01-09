import pickle
import click
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- 1. FONCTIONS UTILITAIRES (Pour ne pas répéter le code) ---
def load_and_split():
    """Charge et prépare les données."""
    data = load_breast_cancer()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler

def save_model(model, scaler, filename):
    """Sauvegarde le modèle et le scaler."""
    with open(filename, "wb") as f:
        pickle.dump((model, scaler), f)
    print(f"💾 Modèle sauvegardé dans {filename}")

# --- 2. CLI PRINCIPALE ---
@click.group()
def app():
    """Application de classification Multi-Modèles."""
    pass

# --- 3. GROUPE TRAIN (Le chef du département Entraînement) ---
@app.group()
def train():
    """Sous-commandes pour entraîner différents modèles."""
    pass

# -> Commande SVM
@train.command()
@click.option("--kernel", default="linear", help="Type de noyau (linear, rbf, poly).")
@click.option("-o", "--output", default="model_svm.pkl", help="Fichier de sortie.")
def svm(kernel, output):
    """Entraîne un Support Vector Machine."""
    x_train, _, y_train, _, scaler = load_and_split()
    
    print(f"💪 Entraînement SVM (Kernel: {kernel})...")
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)
    
    save_model(model, scaler, output)

# -> Commande KNN
@train.command()
@click.option("--neighbors", "-k", default=5, help="Nombre de voisins.")
@click.option("-o", "--output", default="model_knn.pkl", help="Fichier de sortie.")
def knn(neighbors, output):
    """Entraîne un K-Nearest Neighbors."""
    x_train, _, y_train, _, scaler = load_and_split()
    
    print(f"💪 Entraînement KNN (Voisins: {neighbors})...")
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(x_train, y_train)
    
    save_model(model, scaler, output)

# --- 4. COMMANDE EVALUATE ---
@app.command()
@click.argument("model_file")
def evaluate(model_file):
    """Charge et évalue un modèle."""
    print(f"📂 Chargement de {model_file}...")
    try:
        with open(model_file, "rb") as f:
            model, scaler = pickle.load(f)
    except FileNotFoundError:
        print("❌ Fichier introuvable.")
        return

    # On a besoin des données de test
    _, x_test, _, y_test, _ = load_and_split()
    # On applique le scaler chargé !
    x_test = scaler.transform(x_test)
    
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Accuracy: {acc:.2f}")

if __name__ == "__main__":
    app()