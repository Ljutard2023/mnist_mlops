# 1. Image de base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 2. Installation des outils système
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Copie des fichiers de config (uv.lock et pyproject.toml sont ici !)
WORKDIR /app
COPY uv.lock pyproject.toml README.md LICENSE /app/

# 4. Installation des dépendances
RUN uv sync --locked --no-cache --no-install-project

# 5. Copie du code source et des données
# On copie le dossier 'src' entier
COPY src/ /app/src/
# On copie le dossier 'data' entier (attention à la taille si tu as de gros fichiers !)
COPY data/ /app/data/

# 6. Lancement de l'entraînement
# Ton script est bien dans src/my_project/train.py d'après ton 'ls'
# On utilise "python -m" (module) pour que les imports relatifs fonctionnent
ENTRYPOINT ["uv", "run", "python", "-m", "my_project.train"]