"""Garantiza que el backend nunca dependa de la GUI.

Regresión de los fallos vistos con el monorepo (ModuleNotFoundError: PySide6
en Colab): ningún fichero de modules/, imkit/ o server.py puede importar
PySide6, app.*, dayu ni PySide. Corre con python3 stdlib, sin dependencias.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN = [ROOT / "modules", ROOT / "imkit", ROOT / "server.py"]
FORBIDDEN = re.compile(r"^\s*(from|import)\s+(PySide6|PySide|app\.|dayu)", re.M)


def iter_files():
    for base in SCAN:
        if base.is_file():
            yield base
        else:
            yield from sorted(base.rglob("*.py"))


def test_no_gui_deps():
    offenders = []
    for f in iter_files():
        if "__pycache__" in f.parts:
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines, 1):
            m = FORBIDDEN.match(line)
            if not m:
                continue
            # Se permite import opcional tras try: (patrón sancionado para
            # degradar sin PySide, ej. language_utils). Obligatorio = fallo.
            context = "\n".join(lines[max(0, n - 4):n - 1])
            if "try:" in context and "ImportError" in "\n".join(lines[n - 1:n + 3]):
                continue
            offenders.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, "imports duros de GUI/app encontrados:\n" + "\n".join(offenders)
