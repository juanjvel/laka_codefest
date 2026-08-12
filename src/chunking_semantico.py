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

Encoder de producción: intfloat/multilingual-e5-small (resultado del
benchmark del equipo). Los modelos de la familia E5 requieren que el texto
se prefije con "query: " (consultas) o "passage: " (documentos/chunks)
antes de codificar; sin este prefijo la calidad del embedding se degrada
notablemente. Este módulo aplica "passage: " en todas las llamadas a
encode() usadas para chunking/indexación. El campo `texto` de la metadata
se guarda SIN el prefijo (es un artefacto de codificación, no contenido
del fragmento); el prefijo debe volver a aplicarse en el futuro paso de
indexación FAISS y, con "query: ", al vector de consulta en recuperación.

Regla de completitud lingüística (Sección 3.3 del spec) frente al límite
de 250 palabras (Sección 9.2.1): cuando una ÚNICA oración real excede
max_palabras por sí sola (frecuente en documentos de política pública con
cláusulas largas), no existe una división que respete ambas reglas de
forma perfecta. Este módulo prioriza no partir literalmente a mitad de
palabra: distingue prosa genuina de artefactos no lingüísticos (tablas o
listas mal segmentadas por el tokenizador de oraciones, comunes en texto
extraído de PDF) mediante `_parece_prosa()`, y para la prosa busca el
separador de cláusula (coma/punto y coma/dos puntos) más cercano al
límite en vez de cortar por bloque de palabras. Cada partición forzada
(clausal o por bloque) se registra en un archivo de auditoría para que el
equipo pueda revisarlas y documentarlas como excepción en el informe
técnico.
"""
import json
import os
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


def _distancias_semanticas(oraciones: list, modelo, buffer_size: int = 1,
                            prefijo_encoder: str = "passage: ") -> np.ndarray:
    """Devuelve la distancia coseno (1 - similitud) entre cada par de
    oraciones consecutivas, usando ventanas de contexto para embeber.

    prefijo_encoder: prefijo requerido por modelos de la familia E5
    (intfloat/multilingual-e5-*) para codificar texto de documento. Se
    aplica solo a la entrada del encoder, nunca se guarda en la metadata."""
    if len(oraciones) < 2:
        return np.array([])
    textos_ventana = [_texto_ventana(oraciones, i, buffer_size) for i in range(len(oraciones))]
    if prefijo_encoder:
        textos_ventana = [f"{prefijo_encoder}{t}" for t in textos_ventana]
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


_CONECTORES_PROSA = {
    # es
    "que", "y", "de", "la", "el", "los", "las", "en", "con", "para", "por",
    "un", "una", "se", "su", "sus", "como", "porque", "aunque", "cuando",
    "entre", "sobre", "sin", "es", "son", "fue", "han", "esta", "este",
    # pt (comparte varios con es)
    "da", "do", "das", "dos", "com", "uma", "num", "essa", "esse",
    # en
    "the", "and", "of", "that", "which", "with", "for", "is", "are", "was",
    "to", "in", "on", "as", "this", "these",
}


def _parece_prosa(texto: str, umbral_densidad: float = 0.12) -> bool:
    """Heurística para decidir si una 'oración' sin puntuación de cierre
    detectada por NLTK es prosa genuina (una oración real que el
    tokenizador no logró delimitar, p.ej. por error de OCR) o un artefacto
    no lingüístico (tabla/lista extraída de PDF, del tipo 'Departamento_1
    poblacion 1037 area 211 ...'). Se basa en la densidad de conectores
    gramaticales comunes (es/pt/en): la prosa real los usa con frecuencia;
    los volcados tabulares de pares etiqueta-valor casi nunca. El umbral
    es conservador y debe verificarse empíricamente contra el corpus real
    durante la prueba de concepto (recomendado: revisar el archivo de
    auditoría de particiones forzadas tras un primer corrido)."""
    palabras = [p.strip(",.;:") .lower() for p in texto.split()]
    palabras = [p for p in palabras if p]
    if not palabras:
        return False
    hits = sum(1 for p in palabras if p in _CONECTORES_PROSA)
    return (hits / len(palabras)) > umbral_densidad


def _punto_de_corte_clausal(palabras: list, max_palabras: int, retroceso_maximo: int = 40) -> int:
    """Busca, retrocediendo desde max_palabras, la palabra más cercana que
    termina en coma, punto y coma o dos puntos, para cortar ahí en vez de
    a mitad de una palabra o idea. Devuelve max_palabras (corte duro) si
    no encuentra ningún separador dentro de la ventana de retroceso."""
    limite_inferior = max(1, max_palabras - retroceso_maximo)
    for i in range(max_palabras - 1, limite_inferior - 1, -1):
        if palabras[i].endswith((",", ";", ":")):
            return i + 1
    return max_palabras


def _dividir_texto_largo(texto: str, max_palabras: int, doc_id: str = "",
                          eventos_particion: list = None) -> list:
    """Último recurso para 'oraciones' patológicamente largas (según NLTK)
    que exceden max_palabras por sí solas. Dos casos:

    1. Artefacto no lingüístico (tabla/lista sin puntuación real, p.ej.
       una fila de presupuesto por línea): no son oraciones genuinas, así
       que cortarlas por bloques de palabras no viola el espíritu de la
       Sección 3.3. Se prioriza cortar por salto de línea (frontera
       natural entre filas) y solo si una línea individual sigue
       excediendo el límite se recurre a bloques de palabras.
    2. Prosa genuina que por sí sola supera max_palabras (cláusulas largas
       comunes en informes de política pública): SÍ es una oración real,
       así que en vez de cortar a mitad de palabra se busca el separador
       de cláusula (coma/punto y coma/dos puntos) más cercano al límite.
       Esto no es un corte oracional perfecto (Sección 3.3 exige límites
       de ORACIÓN, no de cláusula), pero es la opción menos mala frente a
       la restricción dura de 250 palabras: se documenta como excepción y
       cada ocurrencia se registra en `eventos_particion` para auditoría.
    """
    lineas = [l.strip() for l in texto.split("\n") if l.strip()] or [texto]
    piezas = []
    for linea in lineas:
        palabras = linea.split()
        if len(palabras) <= max_palabras:
            piezas.append(linea)
            continue

        es_prosa = _parece_prosa(linea)
        if es_prosa:
            resto = palabras
            while len(resto) > max_palabras:
                corte = _punto_de_corte_clausal(resto, max_palabras)
                piezas.append(" ".join(resto[:corte]))
                resto = resto[corte:]
            if resto:
                piezas.append(" ".join(resto))
        else:
            for i in range(0, len(palabras), max_palabras):
                piezas.append(" ".join(palabras[i:i + max_palabras]))

        if eventos_particion is not None:
            eventos_particion.append({
                "doc_id": doc_id,
                "tipo": "clausal" if es_prosa else "bloque",
                "num_palabras_original": len(palabras),
                "vista_previa": linea[:120],
            })
    return piezas


def _forzar_limite_palabras(oraciones_grupo: list, max_palabras: int, overlap_oraciones: int,
                             doc_id: str = "", eventos_particion: list = None) -> list:
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
            # La "oración" que devolvió NLTK ya excede el límite por sí
            # sola: puede ser prosa genuina o un artefacto no lingüístico
            # (ver _dividir_texto_largo, que distingue ambos casos).
            if chunk_actual:
                fragmentos.append(" ".join(chunk_actual))
                chunk_actual, palabras_actuales = [], 0
            fragmentos.extend(_dividir_texto_largo(
                oracion, max_palabras, doc_id=doc_id, eventos_particion=eventos_particion
            ))
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
    prefijo_encoder: str = "passage: ",
    eventos_particion: list = None,
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
    prefijo_encoder: prefijo que exige el encoder para texto de documento
        (E5: "passage: "). Pasar "" si se cambia a un encoder que no lo
        requiera.
    eventos_particion: lista opcional (mutable) donde se registran los
        casos en que una oración única excedió max_palabras y tuvo que
        forzarse una partición (clausal o por bloque de palabras), para
        auditoría posterior.

    Devuelve una lista de strings (texto de cada chunk).
    """
    texto = doc.get("texto_limpio", "")
    if not texto:
        return []

    doc_id = doc.get("doc_id", "DOC-UNKNOWN")
    oraciones = tokenizar_oraciones(texto, doc.get("idioma_detectado", ""))
    if not oraciones:
        return []
    if len(oraciones) == 1:
        grupos = [oraciones]
    else:
        distancias = _distancias_semanticas(oraciones, modelo, buffer_size, prefijo_encoder)
        indices_de_corte = _detectar_indices_de_corte(distancias, percentil_breakpoint)
        grupos = _agrupar_por_cortes(oraciones, indices_de_corte)
        grupos = _fusionar_grupos_pequenos(grupos, min_palabras, max_palabras)

    fragmentos = []
    for grupo in grupos:
        if sum(contar_palabras(o) for o in grupo) > max_palabras:
            fragmentos.extend(_forzar_limite_palabras(
                grupo, max_palabras, overlap_oraciones_fallback,
                doc_id=doc_id, eventos_particion=eventos_particion,
            ))
        else:
            fragmentos.append(" ".join(grupo))
    return fragmentos


def _obtener_tokenizer(modelo):
    """Extrae el tokenizer HuggingFace subyacente de un SentenceTransformer
    ya cargado, para calcular num_tokens de forma exacta (no aproximada
    por conteo de palabras)."""
    tok = getattr(modelo, "tokenizer", None)
    if tok is not None:
        return tok
    try:
        return modelo._first_module().tokenizer
    except Exception:
        return None


def construir_metadata_chunks(doc: dict, textos_fragmentos: list, tokenizer=None,
                               max_tokens_encoder: int = 512,
                               eventos_particion: list = None) -> list:
    """Arma los registros de metadata obligatoria (Tabla 1 del spec) para
    los fragmentos de un documento.

    tokenizer: tokenizer real del encoder (ver _obtener_tokenizer). Si es
        None, num_tokens cae a un proxy por conteo de palabras (menos
        preciso: el multilingual-e5-small usa subword tokenization, así
        que 250 palabras en español no equivalen a 250 tokens).
    max_tokens_encoder: ventana máxima del encoder (512 para
        multilingual-e5-small). Si un chunk la excede pese a respetar las
        250 palabras, se registra en eventos_particion para revisión
        manual en vez de fallar silenciosamente en la etapa de indexación.
    """
    doc_id = doc.get("doc_id", "DOC-UNKNOWN")
    registros = []
    for posicion, texto_chunk in enumerate(textos_fragmentos):
        if tokenizer is not None:
            num_tokens = len(tokenizer.encode(texto_chunk, add_special_tokens=True))
        else:
            num_tokens = contar_palabras(texto_chunk)

        if tokenizer is not None and num_tokens > max_tokens_encoder and eventos_particion is not None:
            eventos_particion.append({
                "doc_id": doc_id,
                "tipo": "excede_ventana_encoder",
                "num_tokens": num_tokens,
                "vista_previa": texto_chunk[:120],
            })

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


def procesar_corpus(
    rutas_entrada: list,
    ruta_metadata_salida: str,
    modelo,
    tokenizer=None,
    ruta_auditoria: str = None,
    ruta_fallidos: str = None,
    **kwargs_fragmentacion,
) -> dict:
    """Procesa uno o más archivos *_documentos.jsonl (salida de
    limpieza_extraccion.py) y escribe un único metadata.jsonl combinado,
    con el esquema obligatorio de la Tabla 1 del spec.

    Un documento fallido (excepción durante chunking) no interrumpe el
    resto del corpus: se registra en ruta_fallidos y se continúa, siguiendo
    el mismo patrón de auditoría _pendientes.jsonl usado en las etapas de
    extracción del proyecto.
    """
    eventos_particion, fallidos = [], []
    total_docs, total_chunks = 0, 0

    with open(ruta_metadata_salida, "w", encoding="utf-8") as f_out:
        for ruta in rutas_entrada:
            if not os.path.exists(ruta):
                print(f"AVISO: no se encontro {ruta}, se omite.")
                continue
            with open(ruta, encoding="utf-8") as f_in:
                for num_linea, linea in enumerate(f_in, start=1):
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        doc = json.loads(linea)
                        fragmentos = fragmentar_documento_semantico(
                            doc, modelo, eventos_particion=eventos_particion,
                            **kwargs_fragmentacion,
                        )
                        metadata_doc = construir_metadata_chunks(
                            doc, fragmentos, tokenizer=tokenizer,
                            eventos_particion=eventos_particion,
                        )
                        for registro in metadata_doc:
                            f_out.write(json.dumps(registro, ensure_ascii=False) + "\n")
                        total_docs += 1
                        total_chunks += len(metadata_doc)
                    except Exception as e:
                        fallidos.append({
                            "archivo": ruta, "linea": num_linea, "error": str(e),
                        })

                    if total_docs and total_docs % 100 == 0:
                        print(f"  ... {total_docs} documentos procesados, {total_chunks} chunks")

    if ruta_auditoria and eventos_particion:
        with open(ruta_auditoria, "w", encoding="utf-8") as f_aud:
            for ev in eventos_particion:
                f_aud.write(json.dumps(ev, ensure_ascii=False) + "\n")

    if ruta_fallidos and fallidos:
        with open(ruta_fallidos, "w", encoding="utf-8") as f_fail:
            for ev in fallidos:
                f_fail.write(json.dumps(ev, ensure_ascii=False) + "\n")

    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "particiones_forzadas": len(eventos_particion),
        "fallidos": len(fallidos),
    }


if __name__ == "__main__":
    import argparse
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description="Chunking semantico F1 + F3 -> metadata.jsonl")
    parser.add_argument("--entrada", nargs="+", default=[
        "../data/clean/f1_documentos.jsonl",
        "../data/clean/f3_documentos.jsonl",
    ], help="Archivos *_documentos.jsonl a procesar (se combinan en un solo metadata.jsonl)")
    parser.add_argument("--salida", default="../data/clean/metadata.jsonl",
                         help="Ruta del metadata.jsonl combinado (Tabla 1 del spec)")
    parser.add_argument("--auditoria", default="../data/clean/_chunking_particiones_forzadas.jsonl")
    parser.add_argument("--fallidos", default="../data/clean/_chunking_pendientes.jsonl")
    parser.add_argument("--modelo", default="intfloat/multilingual-e5-small")
    parser.add_argument("--max-palabras", type=int, default=250)
    parser.add_argument("--min-palabras", type=int, default=40)
    parser.add_argument("--percentil-breakpoint", type=float, default=95.0)
    args = parser.parse_args()

    asegurar_recursos_nltk()

    print(f"Cargando encoder: {args.modelo} ...")
    modelo = SentenceTransformer(args.modelo)
    tokenizer = _obtener_tokenizer(modelo)
    if tokenizer is None:
        print("AVISO: no se pudo obtener el tokenizer del modelo; "
              "num_tokens usara conteo de palabras como proxy.")

    resumen = procesar_corpus(
        rutas_entrada=args.entrada,
        ruta_metadata_salida=args.salida,
        modelo=modelo,
        tokenizer=tokenizer,
        ruta_auditoria=args.auditoria,
        ruta_fallidos=args.fallidos,
        max_palabras=args.max_palabras,
        min_palabras=args.min_palabras,
        percentil_breakpoint=args.percentil_breakpoint,
    )

    print(f"\nDocumentos procesados: {resumen['total_docs']}")
    print(f"Chunks generados: {resumen['total_chunks']}")
    print(f"Metadata escrita en: {args.salida}")
    if resumen["particiones_forzadas"]:
        print(f"AVISO: {resumen['particiones_forzadas']} chunks requirieron particion forzada "
              f"-> revisar {args.auditoria} y documentar como excepcion en el informe tecnico")
    if resumen["fallidos"]:
        print(f"AVISO: {resumen['fallidos']} documentos fallaron durante el chunking "
              f"-> revisar {args.fallidos}")
