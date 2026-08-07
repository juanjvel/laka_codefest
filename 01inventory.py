from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

FENOMENO = 3
SUBFENOMENO = "F3-B"
REPO_ROOT = Path(__file__).resolve().parent
RAW_ROOT = REPO_ROOT
EXT_TO_FORMATO = {".pdf": "pdf", ".json": "json", ".csv": "csv", ".xlsx": "xlsx", ".xls": "xlsx", ".txt": "txt", ".md": "txt", ".html": "html", ".htm": "html", ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image", ".avif": "image", ".pbf": "pbf"}
IGNORAR_NOMBRES = {".DS_Store", "Thumbs.db"}
CARPETAS_EXCLUIDAS = {"Alertas_Tempranas", "Amazon_Underworld"}

@dataclass(frozen=True)
class Documento:
    doc_id: str
    path: Path
    fuente: str
    formato: str
    fenomeno: int = FENOMENO
    subfenomeno: str = SUBFENOMENO

@dataclass(frozen=True)
class Excluido:
    fuente: str
    motivo: str
    fenomeno: int = FENOMENO
    subfenomeno: str = SUBFENOMENO

def crear_doc_id(path: Path, raiz: Path) -> str:

    relpath = path.relative_to(raiz).as_posix()
    digest = hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:12]
    return f"F3B-{digest}"

def _carpeta_excluida(path: Path, raiz: Path) -> str | None:

    relpath = path.relative_to(raiz)
    for parte in relpath.parts[:-1]:
        if parte in CARPETAS_EXCLUIDAS:
            return parte
    return None

def iter_documentos(raiz: Path = RAW_ROOT) -> Iterator[Documento]:

    if not raiz.exists():
        raise FileNotFoundError(f"No existe la carpeta de fuentes: {raiz}")

    for path in sorted(raiz.rglob("*")):
        if not path.is_file():
            continue
        if path.name in IGNORAR_NOMBRES:
            continue
        if _carpeta_excluida(path, raiz) is not None:
            continue
        formato = EXT_TO_FORMATO.get(path.suffix.lower())
        # Extensión no soportada:
        # se ignora aquí, pero queda registrada en iter_excluidos.
        if formato is None:
            continue
        relpath = path.relative_to(raiz).as_posix()
        yield Documento(doc_id=crear_doc_id(path, raiz), path=path, fuente=relpath, formato=formato)

def iter_excluidos(raiz: Path = RAW_ROOT) -> Iterator[Excluido]:

    if not raiz.exists():
        raise FileNotFoundError(f"No existe la carpeta de fuentes: {raiz}")

    for path in sorted(raiz.rglob("*")):
        if not path.is_file():
            continue
        relpath = path.relative_to(raiz).as_posix()
        if path.name in IGNORAR_NOMBRES:
            yield Excluido(fuente=relpath, motivo="archivo_sistema_ignorado")
            continue
        carpeta = _carpeta_excluida(path, raiz)
        if carpeta is not None:
            yield Excluido(fuente=relpath, motivo=f"carpeta_fuera_de_alcance:{carpeta}")
            continue
        formato = EXT_TO_FORMATO.get(path.suffix.lower())
        if formato is None:
            extension = path.suffix.lower() or "sin_extension"
            yield Excluido(fuente=relpath, motivo=f"extension_no_soportada:{extension}")

if __name__ == "__main__":
    documentos = list(iter_documentos())
    excluidos = list(iter_excluidos())
    print(f"Documentos encontrados : {len(documentos)}")
    print(f"Excluidos              : {len(excluidos)}")

    conteo = {}
    for documento in documentos:
        conteo[documento.formato] = (conteo.get(documento.formato, 0) + 1)

    for formato, cantidad in sorted(conteo.items()):
        print(f"  {formato}: {cantidad}")