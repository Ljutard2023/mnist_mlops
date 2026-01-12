FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# On n'oublie pas les fichiers de licence et readme cette fois !
COPY uv.lock pyproject.toml README.md LICENSE /app/

RUN uv sync --locked --no-cache --no-install-project

COPY src/ /app/src/
COPY data/ /app/data/

# La seule différence est ici : on lance 'evaluate'
ENTRYPOINT ["uv", "run", "python", "-m", "my_project.evaluate"]