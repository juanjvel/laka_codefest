from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Any, Optional

try:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
except ImportError:
    DetectorFactory = None
    LangDetectException = Exception
    detect = None

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
PAGE_NUMBER_RE = re.compile(r"^\s*(?:p[aá]gina\s*)?(?:\d{1,5})(?:\s*(?:de|/)\s*\d{1,5})?\s*$", re.IGNORECASE)

def to_utf8(texto: str) -> str:
    """
    Normalización Unicode NFC.
    """
    return unicodedata.normalize("NFC", texto)

def normalize_whitespace(texto: str) -> str:
    texto = CONTROL_CHARS_RE.sub("", texto)
    texto = texto.replace("\r\n", "\n")
    texto = texto.replace("\r", "\n")
    texto = MULTI_SPACE_RE.sub(" ", texto)

    lineas = []
    for linea in texto.split("\n"):
        linea = linea.strip()
        lineas.append(linea)
    texto = "\n".join(lineas)

    texto = MULTI_BLANK_LINES_RE.sub("\n\n", texto)

    return texto.strip()

def _lineas_no_vacias(pagina: str) -> list[str]:
    return [linea.strip() for linea in pagina.split("\n") if linea.strip()]

def _normalizar_para_comparar(linea: str) -> str:
    linea = linea.strip()
    linea = re.sub(r"\s+", " ", linea)
    return linea.casefold()

def strip_repeated_pdf_boilerplate(paginas: list[str], min_repeticiones: int = 3) -> list[str]:

    if len(paginas) < min_repeticiones:
        return paginas

    conteos = Counter()
    posiciones: dict[str, list[str]] = {}

    for pagina in paginas:
        lineas = _lineas_no_vacias(pagina)
        if not lineas:
            continue
        cantidad = len(lineas)
        for indice, linea in enumerate(lineas):
            normalizada = _normalizar_para_comparar(linea)
            if not normalizada:
                continue
            conteos[normalizada] += 1
            posicion = "top" if indice < 3 else "bottom" if indice >= cantidad - 3 else "middle"
            posiciones.setdefault(normalizada, []).append(posicion)

    candidatos = set()
    for linea, cantidad in conteos.items():
        if cantidad < min_repeticiones:
            continue

        if PAGE_NUMBER_RE.match(linea):
            candidatos.add(linea)
            continue

        ubicaciones = posiciones.get(linea, [])
        extremos = sum(1 for posicion in ubicaciones if posicion in ("top", "bottom"))

        if extremos >= min_repeticiones:
            if len(linea) <= 180:
                candidatos.add(linea)

    resultado = []
    for pagina in paginas:
        nuevas_lineas = []
        for linea in pagina.split("\n"):
            normalizada = _normalizar_para_comparar(linea)
            if normalizada in candidatos:
                continue
            nuevas_lineas.append(linea)
        resultado.append("\n".join(nuevas_lineas))

    return resultado

def detectar_idioma(texto: str, sample_chars: int = 5000) -> Optional[str]:

    if detect is None:
        return None

    muestra = texto[:sample_chars].strip()
    if len(muestra) < 20:
        return None

    try:
        return detect(muestra)
    except LangDetectException:
        return None
    except Exception:
        return None

def clean_record(raw: dict[str, Any]) -> dict[str, Any]:
    record = dict(raw)
    paginas = record.pop("_paginas", None)
    if paginas is not None:
        paginas = strip_repeated_pdf_boilerplate(paginas)
        record["texto"] = "\n\n".join(paginas)

    texto = record.get("texto", "")
    if texto is None:
        texto = ""
    texto = str(texto)
    texto = to_utf8(texto)
    texto = normalize_whitespace(texto)
    record["texto"] = texto
    record["idioma"] = detectar_idioma(texto)

    return record