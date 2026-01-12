import os
from invoke import task

# On récupère le dossier courant pour les volumes Docker (équivalent de $(pwd))
CURRENT_DIR = os.getcwd()

@task
def build_docker(ctx):
    """
    Construit les images Docker pour l'entraînement et l'évaluation.
    """
    print("🏗️  Construction de l'image d'entraînement (train:latest)...")
    # Note: On précise bien le chemin 'dockerfiles/train.dockerfile'
    ctx.run("docker build -f dockerfiles/train.dockerfile . -t train:latest")
    
    print("🏗️  Construction de l'image d'évaluation (evaluate:latest)...")
    ctx.run("docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest")
    
    print("✅ Toutes les images sont prêtes !")

@task
def train_docker(ctx):
    """
    Lance l'entraînement DANS le conteneur Docker (avec sauvegarde locale).
    """
    print("🚀 Lancement du conteneur d'entraînement...")
    
    # On monte les volumes pour récupérer le modèle et les rapports
    # equivalent de : -v $(pwd)/models:/app/models
    volumes = f"-v {CURRENT_DIR}/models:/app/models -v {CURRENT_DIR}/reports:/app/reports"
    
    ctx.run(f"docker run --rm {volumes} train:latest")

@task
def evaluate_docker(ctx, model_path="models/model.pth"):
    """
    Lance l'évaluation DANS le conteneur Docker.
    """
    print(f"📊 Évaluation du modèle : {model_path}")
    
    # On a besoin d'accéder au dossier models
    volumes = f"-v {CURRENT_DIR}/models:/app/models"
    
    # On passe le chemin du modèle en argument au conteneur
    ctx.run(f"docker run --rm {volumes} evaluate:latest {model_path}")

# --- Tes anciennes tâches (Git, etc.) peuvent rester ici ---
@task
def test(ctx):
    ctx.run("uv run pytest tests/")

@task(pre=[test], help={'message': 'Message du commit'})
def git(ctx, message="Update"):
    ctx.run("git add .")
    ctx.run(f'git commit -m "{message}"')
    ctx.run("git push")