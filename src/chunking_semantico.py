"""
Motor de chunking semántico (Sección 3.2 del spec: "Semántica con superposición").

A diferencia del chunking por tamaño fijo (empaquetar oraciones hasta un
límite de palabras), este motor detecta los puntos de corte a partir de
caídas de similitud coseno entre oraciones consecutivas: agrupa oraciones
que hablan del mismo tema y corta donde el tema cambia. El límite de 250
palabras y la completitud lingüística (Sección 3.3) se siguen respetando
como restricciones duras sobre el resultado del agrupamiento semántico.

Requiere un encoder ya instanciado (SentenceTransformer) para embeber las
oraciones. Debe ser el MISMO encoder que luego indexa los chunks, para que
la frontera semántica se calcule en el mismo espacio vectorial que se usará
en la recuperación.
"""
import re

import numpy as np
import nltk

# Mapa de código de idioma (langdetect, campo idioma_detectado) -> nombre de
# modelo punkt de NLTK. Los idiomas no cubiertos por punkt caen a 'english',
# que sigue siendo un tokenizador razonable basado en puntuación.
IDIOMA_A_PUNKT = {
    "en": "english", "es": "spanish", "pt": "portuguese", "fr": "french",
    "de": "german", "it": "italian", "ru": "russian", "nl": "dutch",
    "pl": "polish", "cs": "czech", "da": "danish", "et": "estonian",
    "fi": "finnish", "el": "greek", "no": "norwegian", "sl": "slovene",
    "sv": "swedish", "tr": "turkish",
}


def asegurar_recursos_nltk():
    for recurso in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(recurso)
        except LookupError:
            nltk.download(recurso.split("/")[-1], quiet=True)


def tokenizar_oraciones(texto: str, idioma_detectado: str = "") -> list:
    """Divide el texto en oraciones respetando fronteras lingüísticas completas."""
    idioma_nltk = IDIOMA_A_PUNKT.get(idioma_detectado, "english")
    try:
        oraciones = nltk.sent_tokenize(texto, language=idioma_nltk)
    except LookupError:
        oraciones = nltk.sent_tokenize(texto, language="english")
    return [o.strip() for o in oraciones if o.strip()]


def contar_palabras(texto: str) -> int:
    return len(texto.split())


def _texto_ventana(oraciones: list, i: int, buffer_size: int) -> str:
    """Combina la oración i con sus vecinas (+/- buffer_size) antes de
    embeberla. Esto suaviza el ruido de embeber oraciones aisladas muy
    cortas y produce fronteras semánticas más estables (técnica estándar
    de chunking semántico por breakpoints, p.ej. LlamaIndex/Kamradt)."""
    inicio = max(0, i - buffer_size)
    fin = min(len(oraciones), i + buffer_size + 1)
    return " ".join(oraciones[inicio:fin])


def _distancias_semanticas(oraciones: list, modelo, buffer_size: int = 1) -> np.ndarray:
    """Devuelve la distancia coseno (1 - similitud) entre cada par de
    oraciones consecutivas, usando ventanas de contexto para embeber."""
    if len(oraciones) < 2:
        return np.array([])
    textos_ventana = [_texto_ventana(oraciones, i, buffer_size) for i in range(len(oraciones))]
    embeddings = modelo.encode(textos_ventana, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    # Vectores normalizados -> producto punto == similitud coseno
    similitudes = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return 1.0 - similitudes


def _detectar_indices_de_corte(distancias: np.ndarray, percentil_breakpoint: float = 95.0) -> list:
    """Un corte se coloca tras la oración i si la distancia semántica hacia
    la oración i+1 supera el percentil `percentil_breakpoint` de todas las
    distancias del documento (umbral adaptativo por documento, no global)."""
    if distancias.size == 0:
        return []
    umbral = np.percentile(distancias, percentil_breakpoint)
    return [i for i, d in enumerate(distancias) if d > umbral]


def _agrupar_por_cortes(oraciones: list, indices_de_corte: list) -> list:
    grupos, inicio = [], 0
    for idx_corte in indices_de_corte:
        grupos.append(oraciones[inicio:idx_corte + 1])
        inicio = idx_corte + 1
    if inicio < len(oraciones):
        grupos.append(oraciones[inicio:])
    return [g for g in grupos if g]


def _fusionar_grupos_pequenos(grupos: list, min_palabras: int, max_palabras: int) -> list:
    """Fusiona grupos semánticos demasiado cortos (ruido/fragmentación
    excesiva) con el grupo siguiente, siempre que no se exceda max_palabras."""
    if not grupos:
        return grupos
    fusionados = [grupos[0]]
    for grupo in grupos[1:]:
        anterior = fusionados[-1]
        palabras_anterior = sum(contar_palabras(o) for o in anterior)
        palabras_grupo = sum(contar_palabras(o) for o in grupo)
        if palabras_anterior < min_palabras and palabras_anterior + palabras_grupo <= max_palabras:
            fusionados[-1] = anterior + grupo
        else:
            fusionados.append(grupo)
    return fusionados


def _dividir_texto_largo(texto: str, max_palabras: int) -> list:
    """Último recurso para 'oraciones' patológicamente largas: texto que
    NLTK devuelve como una sola oración porque no tiene puntuación real,
    típicamente tablas o listas extraídas de PDF (p.ej. tablas de
    presupuesto con una fila por línea y sin punto final). Como no son
    oraciones lingüísticas genuinas, cortarlas no viola el espíritu de la
    Sección 3.3; se prioriza cortar por salto de línea (frontera natural
    entre filas/ítems) y solo si una línea individual sigue excediendo el
    límite se recurre a bloques de palabras."""
    lineas = [l.strip() for l in texto.split("\n") if l.strip()] or [texto]
    piezas = []
    for linea in lineas:
        palabras = linea.split()
        if len(palabras) <= max_palabras:
            piezas.append(linea)
        else:
            for i in range(0, len(palabras), max_palabras):
                piezas.append(" ".join(palabras[i:i + max_palabras]))
    return piezas


def _forzar_limite_palabras(oraciones_grupo: list, max_palabras: int, overlap_oraciones: int) -> list:
    """Fallback obligatorio: si un grupo semántico excede max_palabras, se
    subdivide empaquetando oraciones (idéntico al chunking por tamaño fijo),
    lo que garantiza que ningún chunk final supere el límite y que los
    cortes sigan cayendo exclusivamente en fronteras oracionales.

    La cola de solapamiento se recorta (o se descarta) cuando, sumada a la
    siguiente oración, excedería max_palabras: retener el overlap completo
    sin este chequeo permite que chunk_actual arranque ya por encima del
    límite antes de añadir la oración actual, produciendo fragmentos que
    superan las 250 palabras exigidas por la Sección 9.3.2 del spec.
    """
    fragmentos, chunk_actual, palabras_actuales = [], [], 0
    for oracion in oraciones_grupo:
        p_oracion = contar_palabras(oracion)

        if p_oracion > max_palabras:
            # No es una oración real partible: la "oración" que devolvió
            # NLTK ya excede el límite por sí sola (ver _dividir_texto_largo).
            if chunk_actual:
                fragmentos.append(" ".join(chunk_actual))
                chunk_actual, palabras_actuales = [], 0
            fragmentos.extend(_dividir_texto_largo(oracion, max_palabras))
            continue

        if chunk_actual and palabras_actuales + p_oracion > max_palabras:
            fragmentos.append(" ".join(chunk_actual))
            cola = chunk_actual[-overlap_oraciones:] if overlap_oraciones > 0 else []
            while cola and sum(contar_palabras(o) for o in cola) + p_oracion > max_palabras:
                cola = cola[1:]
            chunk_actual = cola
            palabras_actuales = sum(contar_palabras(o) for o in chunk_actual)

        chunk_actual.append(oracion)
        palabras_actuales += p_oracion
    if chunk_actual:
        fragmentos.append(" ".join(chunk_actual))
    return fragmentos


def fragmentar_documento_semantico(
    doc: dict,
    modelo,
    max_palabras: int = 250,
    min_palabras: int = 40,
    percentil_breakpoint: float = 95.0,
    buffer_size: int = 1,
    overlap_oraciones_fallback: int = 1,
) -> list:
    """
    Fragmenta un documento del corpus (con campos texto_limpio e
    idioma_detectado) en chunks semánticamente cohesivos.

    Parámetros
    ----------
    modelo: SentenceTransformer ya cargado. Debe ser el mismo encoder que
        indexará los chunks resultantes, para que la frontera semántica y
        la búsqueda vivan en el mismo espacio vectorial.
    max_palabras: límite duro por fragmento (Sección 9.2.1 del spec).
    min_palabras: por debajo de este umbral, un grupo semántico se fusiona
        con el siguiente en vez de quedar como chunk aislado.
    percentil_breakpoint: percentil de distancia semántica (por documento)
        a partir del cual se considera que hay un cambio de tema.
    buffer_size: número de oraciones vecinas usadas como contexto al
        embeber cada oración para el cálculo de distancias.
    overlap_oraciones_fallback: solapamiento (en oraciones) usado solo
        cuando un grupo semántico debe subdividirse por exceder max_palabras.

    Devuelve una lista de strings (texto de cada chunk).
    """
    texto = doc.get("texto_limpio", "")
    if not texto:
        return []

    oraciones = tokenizar_oraciones(texto, doc.get("idioma_detectado", ""))
    if not oraciones:
        return []
    if len(oraciones) == 1:
        grupos = [oraciones]
    else:
        distancias = _distancias_semanticas(oraciones, modelo, buffer_size)
        indices_de_corte = _detectar_indices_de_corte(distancias, percentil_breakpoint)
        grupos = _agrupar_por_cortes(oraciones, indices_de_corte)
        grupos = _fusionar_grupos_pequenos(grupos, min_palabras, max_palabras)

    fragmentos = []
    for grupo in grupos:
        if sum(contar_palabras(o) for o in grupo) > max_palabras:
            fragmentos.extend(_forzar_limite_palabras(grupo, max_palabras, overlap_oraciones_fallback))
        else:
            fragmentos.append(" ".join(grupo))
    return fragmentos


def construir_metadata_chunks(doc: dict, textos_fragmentos: list, tokenizer=None) -> list:
    """Arma los registros de metadata obligatoria (Tabla 1 del spec) para
    los fragmentos de un documento."""
    doc_id = doc.get("doc_id", "DOC-UNKNOWN")
    registros = []
    for posicion, texto_chunk in enumerate(textos_fragmentos):
        if tokenizer is not None:
            num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=True))
        else:
            num_tokens = contar_palabras(texto_chunk)
        registros.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}-chunk-{posicion:03d}",
            "fuente": doc.get("fuente", ""),
            "formato": doc.get("formato", ""),
            "fenomeno": doc.get("fenomeno", ""),
            "posicion": posicion,
            "num_tokens": num_tokens,
            "texto": texto_chunk,
        })
    return registros


if __name__ == "__main__":
    import json
    from sentence_transformers import SentenceTransformer

    asegurar_recursos_nltk()

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Cargando encoder de demostración: {MODEL_NAME} ...")
    modelo_demo = SentenceTransformer(MODEL_NAME)

    ruta_muestra = "../data/clean/f2_documentos.jsonl"
    with open(ruta_muestra, encoding="utf-8") as f:
        doc_demo = json.loads(f.readline())

    fragmentos = fragmentar_documento_semantico(doc_demo, modelo_demo)
    metadata = construir_metadata_chunks(doc_demo, fragmentos)

    print(f"\nDocumento: {doc_demo['doc_id']} ({contar_palabras(doc_demo['texto_limpio'])} palabras)")
    print(f"Chunks semánticos generados: {len(metadata)}\n")
    for m in metadata:
        print(f"[{m['chunk_id']}] ({m['num_tokens']} palabras) {m['texto'][:160]}...")
