"""Convierte un script con celdas `# %%` en un notebook .ipynb y lo ejecuta en kernel limpio.

Uso: python -m src.py2nb notebooks/_fuente/01_eda.py
Las celdas `# %% [markdown]` llevan el texto como comentarios (`# `).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def convertir(fuente: Path) -> Path:
    celdas, tipo, buf = [], None, []

    def cerrar() -> None:
        texto = "\n".join(buf).strip("\n")
        if tipo is None or not texto:
            return
        if tipo == "md":
            texto = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in texto.split("\n"))
            celdas.append(new_markdown_cell(texto))
        else:
            celdas.append(new_code_cell(texto))

    for linea in fuente.read_text(encoding="utf-8").split("\n"):
        if linea.startswith("# %%"):
            cerrar()
            tipo, buf = ("md" if "[markdown]" in linea else "code"), []
        else:
            buf.append(linea)
    cerrar()

    nb = new_notebook(cells=celdas, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    destino = fuente.parents[1] / f"{fuente.stem}.ipynb"
    # El kernel arranca en la raíz del proyecto para que `import src` funcione.
    NotebookClient(nb, timeout=3600, resources={"metadata": {"path": str(fuente.parents[2])}}).execute()
    nbformat.write(nb, destino)
    return destino


if __name__ == "__main__":
    print(convertir(Path(sys.argv[1]).resolve()))
