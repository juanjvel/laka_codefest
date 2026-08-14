#!/usr/bin/env python3
"""
generador.py

Módulo de Recuperación (Sección 8 del spec CODEFEST AD ASTRA 2026 - Etapa 1).
Lee las 50 consultas de evaluación, busca en el índice FAISS ya construido,
agrega los fragmentos a nivel de documento (Sección 8.6) mediante max
pooling, aplica el límite de 250 palabras por fragmento (Sección 9.2.1) y
escribe resultados.jsonl en el formato exacto de la Sección 9.3.

Restricción de la Sección 8.3: en ningún punto se usa un modelo generativo.
Toda la lógica opera sobre el vector de consulta, las puntuaciones que
devuelve FAISS y los campos de metadata -- nada de reranking, resúmenes
ni expansión de consulta con un LLM.

Uso:
    python generador.py \
        --indice base_vectorial/encoder_multilingual-e5-small \
        --consultas consultas.jsonl \
        --salida resultados.jsonl

Formato del archivo de consultas (--consultas), autodetectado por extensión:
    .jsonl : una línea por consulta -> {"query_id": "q001", "query": "..."}
    .json  : lista de objetos con las mismas claves, o {"q001": "texto", ...}
    .csv / .xlsx : columnas de id y texto (acepta query_id/id/qid y
                   query/consulta/pregunta/texto como nombres de columna)

Si tu archivo real de consultas usa nombres de columna distintos a los de
ALIAS_ID/ALIAS_TEXTO, agrégalos a esas listas -- es el único punto de
ajuste que debería hacer falta.
"""
import argparse
import json
import linecache
from collections import defaultdict
from pathlib import Path

import faiss
import nltk
from sentence_transformers import SentenceTransformer

MODELO_POR_DEFECTO = "intfloat/multilingual-e5-small"
PREFIJO_QUERY = "query: "

ALIAS_ID = ["query_id", "id", "consulta_id", "qid"]
ALIAS_TEXTO = ["query", "consulta", "pregunta", "texto", "text"]


# ---------------------------------------------------------------------------
# Carga de consultas
# ---------------------------------------------------------------------------
def _normalizar_registro(obj):
    qid = next((obj[k] for k in ALIAS_ID if k in obj), None)
    texto = next((obj[k] for k in ALIAS_TEXTO if k in obj), None)
    if qid is None or texto is None:
        raise ValueError(
            f"registro de consulta sin campos reconocibles (esperaba uno de "
            f"{ALIAS_ID} y uno de {ALIAS_TEXTO}): {obj}"
        )
    return {"query_id": str(qid), "query": str(texto)}


def _consultas_desde_dataframe(df):
    cols = {c.lower(): c for c in df.columns}
    col_id = next((cols[a] for a in ALIAS_ID if a in cols), None)
    col_texto = next((cols[a] for a in ALIAS_TEXTO if a in cols), None)
    if col_id is None or col_texto is None:
        raise ValueError(f"no se encontraron columnas de id/texto en: {list(df.columns)}")
    return [{"query_id": str(r[col_id]), "query": str(r[col_texto])} for _, r in df.iterrows()]


def cargar_consultas(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".jsonl":
        consultas = []
        with open(path, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    consultas.append(_normalizar_registro(json.loads(linea)))
    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [{"query_id": k, "query": v} for k, v in data.items()]
        consultas = [_normalizar_registro(obj) for obj in data]
    elif ext == ".csv":
        import pandas as pd
        consultas = _consultas_desde_dataframe(pd.read_csv(path, sep=None, engine="python"))
    elif ext in (".xlsx", ".xls"):
        import pandas as pd
        consultas = _consultas_desde_dataframe(pd.read_excel(path))
    else:
        raise ValueError(f"formato de archivo de consultas no soportado: {ext}")

    consultas.sort(key=lambda c: c["query_id"])  # garantiza orden q001..q050 (Sección 10.3)
    if len(consultas) != 50:
        print(f"[aviso] se esperaban 50 consultas (q001-q050), se cargaron {len(consultas)}")
    return consultas


# ---------------------------------------------------------------------------
# Fragmentación a <= max_palabras respetando completitud lingüística (9.2.1)
# ---------------------------------------------------------------------------
def dividir_en_subfragmentos(texto, max_palabras=250):
    """Empaqueta oraciones completas en piezas de <= max_palabras.

    Nunca corta una oración a la mitad. Si una sola oración ya supera
    max_palabras (caso raro, p.ej. una fila de CSV sin puntuación), se
    deja entera -- no hay forma de dividirla sin violar la Sección 3.3 /
    9.2.1, así que se prioriza completitud lingüística sobre el límite.
    """
    oraciones = nltk.sent_tokenize(texto)
    piezas, actual, actual_palabras = [], [], 0

    for oracion in oraciones:
        n = len(oracion.split())
        if actual and actual_palabras + n > max_palabras:
            piezas.append(" ".join(actual).strip())
            actual, actual_palabras = [], 0
        actual.append(oracion)
        actual_palabras += n

    if actual:
        piezas.append(" ".join(actual).strip())
    return piezas


# ---------------------------------------------------------------------------
# Agregación a nivel de documento (Sección 8.6)
# ---------------------------------------------------------------------------
def agregar_a_documentos(candidatos, objetos, top_n=3, metodo="max"):
    """Agrupa los chunks candidatos por doc_id y calcula un score agregado
    por documento. "max", "sum" y "mean" son las tres estrategias que
    nombra el spec textualmente (Sección 8.6); por defecto se usa "max"
    (score del fragmento más relevante de cada documento).

    Esta agregación solo afecta el nivel documento, nunca el ranking de
    fragmentos (Sección 9.2): la relevancia de un fragmento es mérito de
    su propio texto, no de cuántos "hermanos" tenga en el documento.
    """
    por_doc = defaultdict(list)
    for score, fid in candidatos:
        por_doc[objetos[fid]["doc_id"]].append(score)

    agregados = []
    for doc_id, scores in por_doc.items():
        if metodo == "max":
            s = max(scores)
        elif metodo == "sum":
            s = sum(scores)
        elif metodo == "mean":
            s = sum(scores) / len(scores)
        else:
            raise ValueError(f"método de agregación desconocido: {metodo}")
        agregados.append((s, doc_id))

    agregados.sort(key=lambda x: -x[0])
    return [doc_id for _, doc_id in agregados[:top_n]]


# ---------------------------------------------------------------------------
# Construcción de la lista de fragmentos (Sección 9.2 / 9.2.1)
# ---------------------------------------------------------------------------
def construir_fragmentos(candidatos, objetos, top_n=10, max_palabras=250):
    fragmentos = []
    for score, fid in candidatos:
        if len(fragmentos) >= top_n:
            break
        obj = objetos[fid]
        texto = obj["texto"]
        piezas = [texto] if len(texto.split()) <= max_palabras else dividir_en_subfragmentos(
            texto, max_palabras)
        for pieza in piezas:
            if len(fragmentos) >= top_n:
                break
            fragmentos.append({
                "rank": len(fragmentos) + 1,
                "chunk_id": obj["chunk_id"],   # mismo chunk_id para todas las sub-piezas (9.2.1)
                "doc_id": obj["doc_id"],
                "text": pieza,
            })
    return fragmentos


# ---------------------------------------------------------------------------
# Recuperación por consulta (Sección 8.1)
# ---------------------------------------------------------------------------
def procesar_consulta(query_id, texto_consulta, index, ruta_metadata, modelo,
                       k_inicial=30, k_maximo=200, top_docs=3, top_fragmentos=10,
                       max_palabras=250, metodo_agregacion="max"):
    q = modelo.encode([PREFIJO_QUERY + texto_consulta], normalize_embeddings=True,
                       convert_to_numpy=True).astype("float32")

    k = min(k_inicial, index.ntotal)
    while True:
        scores, ids = index.search(q, k)
        candidatos = [(s, int(i)) for s, i in zip(scores[0], ids[0]) if i != -1]
        objetos = {fid: json.loads(linecache.getline(str(ruta_metadata), fid + 1))
                   for _, fid in candidatos}

        docs = agregar_a_documentos(candidatos, objetos, top_docs, metodo_agregacion)
        fragmentos = construir_fragmentos(candidatos, objetos, top_fragmentos, max_palabras)

        if len(docs) >= top_docs and len(fragmentos) >= top_fragmentos:
            break
        if k >= k_maximo or k >= index.ntotal:
            print(f"[aviso] {query_id}: no se alcanzaron {top_docs} docs / {top_fragmentos} "
                  f"fragmentos ni con k={k} (docs={len(docs)}, fragmentos={len(fragmentos)})")
            break
        k = min(k * 2, k_maximo, index.ntotal)

    return {
        "query_id": query_id,
        "documents": [{"rank": r + 1, "doc_id": d} for r, d in enumerate(docs[:top_docs])],
        "fragments": fragmentos[:top_fragmentos],
    }


# ---------------------------------------------------------------------------
# Validación del archivo de salida (autodefensa antes de entregar)
# ---------------------------------------------------------------------------
def validar_resultados(ruta_salida, n_esperado=50, max_palabras=250):
    print("\n== Validación de resultados.jsonl ==")
    ok = True
    with open(ruta_salida, encoding="utf-8") as f:
        lineas = [json.loads(l) for l in f if l.strip()]

    if len(lineas) != n_esperado:
        print(f"  [FALLA] {len(lineas)} líneas, se esperaban {n_esperado}")
        ok = False
    else:
        print(f"  [ok] {len(lineas)} líneas")

    ids_vistos = set()
    for obj in lineas:
        qid = obj.get("query_id")
        ids_vistos.add(qid)
        docs = obj.get("documents", [])
        frags = obj.get("fragments", [])
        if len(docs) != 3:
            print(f"  [FALLA] {qid}: {len(docs)} documentos (se esperaban 3)")
            ok = False
        if len(frags) != 10:
            print(f"  [FALLA] {qid}: {len(frags)} fragmentos (se esperaban 10)")
            ok = False
        for fr in frags:
            n_pal = len(fr.get("text", "").split())
            if n_pal > max_palabras:
                print(f"  [FALLA] {qid}: fragmento rank={fr.get('rank')} tiene {n_pal} "
                      f"palabras (> {max_palabras})")
                ok = False

    if len(ids_vistos) != len(lineas):
        print(f"  [FALLA] hay query_id repetidos ({len(lineas)} líneas, {len(ids_vistos)} ids únicos)")
        ok = False

    print("  RESULTADO:", "todo OK" if ok else "hay fallas -- revisa arriba")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indice", required=True, help="carpeta encoder_.../ con index.faiss y metadata.jsonl")
    ap.add_argument("--consultas", required=True, help="archivo con las 50 consultas")
    ap.add_argument("--salida", default="resultados.jsonl")
    ap.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k-inicial", type=int, default=30)
    ap.add_argument("--k-maximo", type=int, default=200)
    ap.add_argument("--agregacion", choices=["max", "sum", "mean"], default="max",
                     help="estrategia de agregación de fragmentos a documento (Sección 8.6)")
    ap.add_argument("--max-palabras", type=int, default=250)
    args = ap.parse_args()

    carpeta = Path(args.indice)
    ruta_metadata = carpeta / "metadata.jsonl"

    print(f"[info] Cargando índice: {carpeta / 'index.faiss'}")
    index = faiss.read_index(str(carpeta / "index.faiss"))

    print(f"[info] Cargando consultas: {args.consultas}")
    consultas = cargar_consultas(args.consultas)

    print(f"[info] Cargando encoder: {args.modelo}")
    kwargs_modelo = {"device": args.device} if args.device else {}
    modelo = SentenceTransformer(args.modelo, **kwargs_modelo)
    print(f"[info] Dispositivo activo: {modelo.device}")

    resultados = []
    for c in consultas:
        resultados.append(procesar_consulta(
            c["query_id"], c["query"], index, ruta_metadata, modelo,
            k_inicial=args.k_inicial, k_maximo=args.k_maximo,
            metodo_agregacion=args.agregacion, max_palabras=args.max_palabras,
        ))

    with open(args.salida, "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] {len(resultados)} consultas procesadas -> {args.salida}")

    validar_resultados(args.salida, n_esperado=len(consultas), max_palabras=args.max_palabras)


if __name__ == "__main__":
    main()
