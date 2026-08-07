from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class ExtractionError(Exception):
    """Error controlado durante la extracción."""
    pass

def ocr(imagen: Any) -> str:
    import pytesseract
    try:
        return pytesseract.image_to_string(imagen) or ""
    except Exception as exc:
        raise ExtractionError(f"Error ejecutando OCR: {exc}") from exc

def extract_pdf(path: Path) -> dict[str, Any]:

    import pdfplumber
    paginas: list[str] = []
    ocr_paginas = 0
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texto = page.extract_text() or ""
                if not texto.strip():
                    try:
                        imagen = page.to_image(resolution=300).original
                        texto = ocr(imagen)
                        ocr_paginas += 1
                    except Exception:
                        # La página permanece vacía,
                        # pero no se detiene todo el PDF.
                        texto = ""
                paginas.append(texto)
    except Exception as exc:
        raise ExtractionError(f"No se pudo procesar PDF: {exc}") from exc

    return {"texto": "\n\n".join(paginas), "_paginas": paginas, "num_paginas": len(paginas), "ocr_paginas": ocr_paginas}

CAMPOS_TEXTO_DIRECTOS = ("body_text", "body", "content", "text", "full_text", "fulltext", "article_text", "article_body", "document_text", "document_body", "description", "abstract", "summary", "excerpt", "extract", "contenido", "contenido_texto", "texto", "texto_completo", "resumen", "descripcion")
CAMPOS_PARRAFOS = ("body_paragraphs", "paragraphs", "parrafos", "paragraph", "sections", "secciones", "blocks", "bloques", "content_blocks")
CAMPOS_METADATA = ("title", "name", "url", "date", "year", "authors", "author", "author_name", "published_at", "publication_date", "source", "publisher", "journal", "tags", "topics", "category", "categories", "doi")

def _limpiar_texto(valor: Any) -> str:

    if valor is None:
        return ""
    if isinstance(valor, str):
        return valor.strip()
    return ""

def _texto_parece_contenido(texto: str) -> bool:
    texto = texto.strip()
    if len(texto) < 30:
        return False
    if texto.startswith(("http://", "https://")) and " " not in texto:
        return False
    return True

def _convertir_lista_texto(valores: list[Any]) -> str:

    partes: list[str] = []
    for valor in valores:
        if valor is None:
            continue
        if isinstance(valor, str):
            texto = valor.strip()
            if texto:
                partes.append(texto)
            continue
        if isinstance(valor, dict):
            encontrado = False
            for campo in ("text", "body", "content", "value", "texto", "contenido", "paragraph", "paragraph_text"):
                if campo in valor:
                    texto = _limpiar_texto(valor[campo])
                    if texto:
                        partes.append(texto)
                        encontrado = True
                        break
            if encontrado:
                continue
            texto = _extraer_texto_recursivo(valor)
            if texto:
                partes.append(texto)
    return "\n\n".join(partes)

def _extraer_texto_recursivo(data: Any, profundidad: int = 0, max_profundidad: int = 30) -> str:
    if profundidad > max_profundidad:
        return ""

    if isinstance(data, str):
        texto = data.strip()
        return texto

    if isinstance(data, list):
        partes: list[str] = []
        for elemento in data:
            texto = _extraer_texto_recursivo(elemento, profundidad + 1, max_profundidad)
            if texto:
                partes.append(texto)
        return "\n\n".join(partes)

    if not isinstance(data, dict):
        return ""

    for campo in CAMPOS_TEXTO_DIRECTOS:
        if campo not in data:
            continue
        valor = data[campo]
        if isinstance(valor, str) and valor.strip():
            return valor.strip()
        if isinstance(valor, list):
            texto = _convertir_lista_texto(valor)
            if texto.strip():
                return texto
        if isinstance(valor, dict):
            texto = _extraer_texto_recursivo(valor, profundidad + 1, max_profundidad)
            if texto.strip():
                return texto

    for campo in CAMPOS_PARRAFOS:
        if campo not in data:
            continue
        valor = data[campo]
        if isinstance(valor, list):
            texto = _convertir_lista_texto(valor)
            if texto.strip():
                return texto
        elif isinstance(valor, str) and valor.strip():
            return valor.strip()
        elif isinstance(valor, dict):
            texto = _extraer_texto_recursivo(valor, profundidad + 1, max_profundidad)
            if texto.strip():
                return texto

    for clave in ("article", "articles", "data", "publication", "publications", "document", "documents", "record", "records", "item", "items", "result", "results", "response", "page", "pages", "entry", "entries", "metadata"):
        if clave not in data:
            continue
        valor = data[clave]
        texto = _extraer_texto_recursivo(valor, profundidad + 1, max_profundidad)
        if texto.strip():
            return texto

    candidatos: list[str] = []
    campos_excluidos = {"id", "_id", "uuid", "slug", "url", "uri", "href", "date", "year", "timestamp", "created_at", "updated_at"}

    for clave, valor in data.items():
        clave_normalizada = str(clave).casefold()
        if clave_normalizada in campos_excluidos:
            continue
        if isinstance(valor, str):
            texto = valor.strip()
            if _texto_parece_contenido(texto):
                candidatos.append(texto)
        elif isinstance(valor, (dict, list)):
            texto = _extraer_texto_recursivo(valor, profundidad + 1, max_profundidad)
            if texto:
                candidatos.append(texto)

    if candidatos:
        return "\n\n".join(candidatos)

    return ""

def _extraer_metadata(data: dict[str, Any]) -> dict[str, Any]:

    metadata: dict[str, Any] = {}
    for campo in CAMPOS_METADATA:
        if campo not in data:
            continue
        valor = data[campo]
        if valor is None:
            continue
        if isinstance(valor, str) and not valor.strip():
            continue
        metadata[campo] = valor
    return metadata

def _buscar_metadata_recursiva(data: Any, resultado: dict[str, Any] | None = None, profundidad: int = 0, max_profundidad: int = 20) -> dict[str, Any]:
    if resultado is None:
        resultado = {}
    if profundidad > max_profundidad:
        return resultado

    if isinstance(data, dict):
        for campo in CAMPOS_METADATA:
            if campo in data and campo not in resultado:
                valor = data[campo]
                if valor is None:
                    continue
                if isinstance(valor, str) and not valor.strip():
                    continue
                resultado[campo] = valor
        for valor in data.values():
            if isinstance(valor, (dict, list)):
                _buscar_metadata_recursiva(valor, resultado, profundidad + 1, max_profundidad)

    elif isinstance(data, list):
        for elemento in data:
            if isinstance(elemento, (dict, list)):
                _buscar_metadata_recursiva(elemento, resultado, profundidad + 1, max_profundidad)

    return resultado

def _extraer_json_objeto(data: dict[str, Any]) -> dict[str, Any]:
    texto = _extraer_texto_recursivo(data)
    metadata = _extraer_metadata(data)

    metadata_recursiva = _buscar_metadata_recursiva(data)
    for clave, valor in metadata_recursiva.items():
        metadata.setdefault(clave, valor)

    if not texto.strip():
        titulo = data.get("title")
        if isinstance(titulo, str) and titulo.strip():
            texto = titulo.strip()

    return {"texto": texto, "estructura_json": "object", **metadata}

def _leer_json_robusto(path: Path) -> Any:
    try:
        contenido = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"No se pudo abrir JSON: {exc}") from exc

    errores: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
        try:
            texto = contenido.decode(encoding)
            return json.loads(texto)
        except UnicodeDecodeError as exc:
            errores.append(f"{encoding}: {exc}")
        except json.JSONDecodeError as exc:
            errores.append(f"{encoding}: {exc}")

    raise ExtractionError("No se pudo interpretar el JSON. " + " | ".join(errores))

def extract_json(path: Path) -> dict[str, Any]:
    data = _leer_json_robusto(path)

    if isinstance(data, list):
        textos: list[str] = []
        metadata_global: dict[str, Any] = {}
        for elemento in data:
            if isinstance(elemento, dict):
                resultado = _extraer_json_objeto(elemento)
                texto = resultado.pop("texto", "")
                if texto.strip():
                    textos.append(texto)
                for clave, valor in resultado.items():
                    metadata_global.setdefault(clave, valor)
            else:
                texto = _extraer_texto_recursivo(elemento)
                if texto.strip():
                    textos.append(texto)

        texto_final = "\n\n".join(textos)
        if not texto_final.strip():
            raise ExtractionError("JSON válido, pero no contiene texto documental recuperable.")

        return {"texto": texto_final, "estructura_json": "list", "num_registros": len(data), **metadata_global}

    if isinstance(data, dict):
        resultado = _extraer_json_objeto(data)
        if not resultado.get("texto", "").strip():
            raise ExtractionError("JSON válido, pero no contiene texto documental recuperable.")
        return resultado

    raise ExtractionError("El JSON no contiene un objeto o lista válida.")

def extract_csv(path: Path) -> dict[str, Any]:
    import pandas as pd
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ExtractionError(f"No se pudo leer CSV: {exc}") from exc

    filas: list[str] = []
    for _, fila in df.iterrows():
        pares: list[str] = []
        for columna, valor in fila.items():
            valor = str(valor).strip()
            if not valor:
                continue
            pares.append(f"{columna}: {valor}")
        if pares:
            filas.append(" | ".join(pares))

    return {"texto": "\n".join(filas), "num_filas": len(df), "num_columnas": len(df.columns)}

def extract_xlsx(path: Path) -> dict[str, Any]:
    import pandas as pd
    try:
        hojas = pd.read_excel(path, sheet_name=None, dtype=str)
    except Exception as exc:
        raise ExtractionError(f"No se pudo leer Excel: {exc}") from exc

    bloques: list[str] = []
    total_filas = 0
    for nombre_hoja, df in hojas.items():
        if df.empty:
            continue
        bloques.append(f"Hoja: {nombre_hoja}")
        for _, fila in df.iterrows():
            pares: list[str] = []
            for columna, valor in fila.items():
                if valor is None:
                    continue
                valor = str(valor).strip()
                if not valor or valor.lower() == "nan":
                    continue
                pares.append(f"{columna}: {valor}")
            if pares:
                bloques.append(" | ".join(pares))
                total_filas += 1

    return {"texto": "\n".join(bloques), "num_filas": total_filas, "num_hojas": len(hojas)}

def extract_txt(path: Path) -> dict[str, Any]:
    try:
        texto = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(f"No se pudo leer TXT/Markdown: {exc}") from exc
    return {"texto": texto}

def extract_html(path: Path) -> dict[str, Any]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ExtractionError("BeautifulSoup no está instalado.") from exc

    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for elemento in soup(["script", "style", "noscript", "svg"]):
            elemento.decompose()
        texto = soup.get_text("\n", strip=True)
    except Exception as exc:
        raise ExtractionError(f"No se pudo procesar HTML: {exc}") from exc

    return {"texto": texto}

def extract_image(path: Path) -> dict[str, Any]:
    from PIL import Image
    try:
        with Image.open(path) as imagen:
            texto = ocr(imagen)
    except Exception as exc:
        raise ExtractionError(f"No se pudo aplicar OCR a imagen: {exc}") from exc
    return {"texto": texto, "ocr": True}

def extract_pbf(path: Path) -> dict[str, Any]:

    try:
        import mapbox_vector_tile
    except ImportError as exc:
        raise ExtractionError("Para procesar PBF se requiere mapbox-vector-tile.") from exc

    try:
        contenido = path.read_bytes()
        capas = mapbox_vector_tile.decode(contenido)
    except Exception as exc:
        raise ExtractionError(f"No se pudo decodificar PBF: {exc}") from exc

    elementos: list[str] = []
    vistos: set[str] = set()
    for nombre_capa, capa in capas.items():
        features = capa.get("features", [])
        for feature in features:
            propiedades = feature.get("properties", {})
            if not isinstance(propiedades, dict):
                continue
            pares: list[str] = []
            for atributo, valor in propiedades.items():
                if valor is None:
                    continue
                texto_valor = str(valor).strip()
                if not texto_valor:
                    continue
                pares.append(f"{atributo}: {texto_valor}")
            if not pares:
                continue
            registro = f"capa: {nombre_capa} | " + " | ".join(pares)
            if registro in vistos:
                continue
            vistos.add(registro)
            elementos.append(registro)

    return {"texto": "\n".join(elementos), "num_elementos": len(elementos)}

HANDLERS = {
    "pdf": extract_pdf,
    "json": extract_json,
    "csv": extract_csv,
    "xlsx": extract_xlsx,
    "txt": extract_txt,
    "html": extract_html,
    "image": extract_image,
    "pbf": extract_pbf,
}

def extraer(documento) -> dict[str, Any]:
    handler = HANDLERS.get(documento.formato)
    if handler is None:
        raise ExtractionError(f"Formato no soportado: {documento.formato}")

    resultado = handler(documento.path)

    texto = resultado.get("texto", "")
    if texto is None:
        texto = ""
    if not isinstance(texto, str):
        texto = str(texto)
    resultado["texto"] = texto

    return resultado