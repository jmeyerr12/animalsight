
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image

# Raiz do projeto
ROOT = Path(__file__).resolve().parent.parent

# Localiza imagens dentro do projeto
extensoes = {".jpg", ".jpeg", ".png", ".bmp"}
ignorar = {".git", ".venv", "results"}

imagens = [
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in extensoes
    and not any(parte in ignorar for parte in p.relative_to(ROOT).parts)
]

# Agrupa imagens pelo nome da pasta da classe
classes = defaultdict(list)

for imagem in imagens:
    classes[imagem.parent.name].append(imagem)

print(f"Total de imagens: {len(imagens)}")

for nome, arquivos in sorted(classes.items()):
    print(f"{nome}: {len(arquivos)} imagens")

if len(classes) != 6:
    print("\nEstrutura diferente da esperada.")
    print("Exemplos de caminhos:", imagens[:5])
    raise SystemExit(1)

# Mostra uma imagem de cada classe
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for ax, (nome, arquivos) in zip(axes.flat, sorted(classes.items())):
    imagem = Image.open(arquivos[0]).convert("L")

    ax.imshow(imagem, cmap="gray")
    ax.set_title(nome)
    ax.axis("off")

plt.tight_layout()
plt.savefig(ROOT / "results" / "amostras_dataset.png", dpi=150)
plt.show()
