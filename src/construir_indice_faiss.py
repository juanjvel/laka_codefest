#!/usr/bin/env python3
"""
construir_indice_faiss.py

Construye el índice vectorial FAISS (encoder intfloat/multilingual-e5-small)
a partir del metadata.jsonl generado por chunking_semantico.py, siguiendo la
Especificación Técnica CODEFEST AD ASTRA 2026 - Etapa 1 (Secciones 4, 5 y 6).

Uso típico (run completo):
    python construir_indice_faiss.py \
        --metadata data/clean/metadata.jsonl \
        --salida base_vectorial/encoder_multilingual-e5-small \
        --batch-size 64

Prueba de concepto rápida (primeros 500 chunks, recomendado antes del run
completo, siguiendo el principio del equipo de validar en ~30 min):
    python construir_indice_faiss.py \
        --metadata data/clean/metadata.jsonl \
        --salida base_vectorial/encoder_multilingual-e5-small_poc \
        --limit 500

Prueba de recuperación al finalizar:
    python construir_indice_faiss.py --metadata ... --salida ... \
        --consulta-prueba "impacto de la IA en la defensa nacional"

Requisitos:
    pip install sentence-transformers faiss-cpu tqdm
    (usar faiss-gpu-cu12 en vez de faiss-cpu si hay GPU NVIDIA disponible)
"""
import argparse
import json
import linecache
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODELO_POR_DEFECTO = "intfloat/multilingual-e5-small"

# La familia E5 exige estos prefijos: sin ellos el espacio vectorial de
# consulta y el de documento no son comparables y la similitud coseno
# se degrada notablemente. El prefijo NO se guarda en el campo "texto" de
# metadata.jsonl (el spec exige texto "sin modificaciones", Tabla 1) —
# se aplica únicamente en el momento de codificar.
PREFIJO_PASSAGE = "passage: "
PREFIJO_QUERY = "query: "

CHECKPOINT_CADA_N_BATCHES = 20  # guarda progreso cada N batches procesados


def contar_lineas(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def iterar_metadata(path, limite=None):
    """Itera metadata.jsonl en orden estricto de archivo.

    El orden de lectura ES el orden de inserción en FAISS, que a su vez
    define el id interno de cada vector (0, 1, 2, ...). Nunca reordenar
    ni paralelizar esta lectura sin preservar el orden: el spec (Sección
    1.4) exige que la línea N de metadata.jsonl corresponda al id interno
    N del índice FAISS.
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, linea in enumerate(f):
            if limite is not None and i >= limite:
                break
            linea = linea.strip()
            if not linea:
                continue
            yield json.loads(linea)


def cargar_checkpoint(ruta_checkpoint):
    if ruta_checkpoint.exists():
        with open(ruta_checkpoint, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"procesados": 0}


def guardar_checkpoint(ruta_checkpoint, procesados):
    with open(ruta_checkpoint, "w", encoding="utf-8") as f:
        json.dump({"procesados": procesados}, f)


def construir_indice(metadata_path, salida_dir, batch_size=64, limite=None,
                      modelo_nombre=MODELO_POR_DEFECTO, reanudar=True, device=None):
    salida_dir = Path(salida_dir)
    salida_dir.mkdir(parents=True, exist_ok=True)

    ruta_index = salida_dir / "index.faiss"
    ruta_metadata_salida = salida_dir / "metadata.jsonl"
    ruta_checkpoint = salida_dir / "_checkpoint.json"

    total = limite if limite else contar_lineas(metadata_path)
    print(f"[info] Chunks a procesar: {total}")

    checkpoint = cargar_checkpoint(ruta_checkpoint) if reanudar else {"procesados": 0}
    inicio = checkpoint["procesados"]

    print(f"[info] Cargando encoder: {modelo_nombre}")
    kwargs_modelo = {"device": device} if device else {}
    modelo = SentenceTransformer(modelo_nombre, **kwargs_modelo)
    dim = modelo.get_sentence_embedding_dimension()
    print(f"[info] Dispositivo activo: {modelo.device}"
          + ("  <- revisa que diga 'cuda', no 'cpu', si esperabas usar GPU"
             if str(modelo.device) == "cpu" else ""))

    if inicio > 0 and ruta_index.exists():
        print(f"[info] Reanudando desde checkpoint: {inicio} vectores ya indexados")
        index = faiss.read_index(str(ruta_index))
        modo_metadata = "a"
    else:
        inicio = 0
        index = faiss.IndexFlatIP(dim)  # producto interno == coseno con vectores normalizados
        modo_metadata = "w"

    f_meta_salida = open(ruta_metadata_salida, modo_metadata, encoding="utf-8")

    batch_textos, batch_objs = [], []
    procesados = inicio
    n_batches_desde_checkpoint = 0
    t0 = time.time()

    generador = iterar_metadata(metadata_path, limite)
    for _ in range(inicio):  # saltar lo ya procesado si reanudamos
        next(generador, None)

    pbar = tqdm(total=total, initial=inicio, desc="Codificando")

    def flush_batch():
        nonlocal batch_textos, batch_objs, procesados, n_batches_desde_checkpoint
        if not batch_textos:
            return
        textos_con_prefijo = [PREFIJO_PASSAGE + t for t in batch_textos]
        vectores = modelo.encode(
            textos_con_prefijo,
            batch_size=len(textos_con_prefijo),
            normalize_embeddings=True,  # clave: habilita coseno vía IndexFlatIP
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        index.add(vectores)
        for obj in batch_objs:
            f_meta_salida.write(json.dumps(obj, ensure_ascii=False) + "\n")

        procesados += len(batch_textos)
        pbar.update(len(batch_textos))
        n_batches_desde_checkpoint += 1
        batch_textos, batch_objs = [], []

        if n_batches_desde_checkpoint >= CHECKPOINT_CADA_N_BATCHES:
            faiss.write_index(index, str(ruta_index))
            guardar_checkpoint(ruta_checkpoint, procesados)
            f_meta_salida.flush()
            n_batches_desde_checkpoint = 0

    try:
        for obj in generador:
            batch_textos.append(obj["texto"])
            batch_objs.append(obj)
            if len(batch_textos) >= batch_size:
                flush_batch()
        flush_batch()  # último batch parcial
    finally:
        faiss.write_index(index, str(ruta_index))
        guardar_checkpoint(ruta_checkpoint, procesados)
        f_meta_salida.close()
        pbar.close()

    dt = time.time() - t0
    print(f"[ok] {procesados} vectores indexados en {dt/60:.1f} min")
    print(f"[ok] index.faiss    -> {ruta_index}  ({index.ntotal} vectores, dim={index.d})")
    print(f"[ok] metadata.jsonl -> {ruta_metadata_salida}")

    # Verificación de alineación obligatoria (Sección 1.4 del spec: el orden
    # de las líneas debe coincidir con los ids internos de FAISS)
    n_meta = contar_lineas(ruta_metadata_salida)
    assert n_meta == index.ntotal, (
        f"[ERROR] Desalineación: metadata.jsonl tiene {n_meta} líneas "
        f"pero el índice tiene {index.ntotal} vectores"
    )
    print("[ok] Alineación metadata.jsonl <-> índice FAISS verificada")

    if ruta_checkpoint.exists():
        ruta_checkpoint.unlink()  # limpieza tras un run completo exitoso


def prueba_consulta(salida_dir, consulta, k=5, modelo_nombre=MODELO_POR_DEFECTO):
    """Prueba rápida de recuperación (Sección 8 del spec: mismo encoder,
    mismo prefijo E5, similitud coseno vía producto interno)."""
    salida_dir = Path(salida_dir)
    ruta_metadata = str(salida_dir / "metadata.jsonl")

    index = faiss.read_index(str(salida_dir / "index.faiss"))
    modelo = SentenceTransformer(modelo_nombre)

    q = modelo.encode([PREFIJO_QUERY + consulta], normalize_embeddings=True,
                       convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q, k)

    print(f"\nConsulta: {consulta!r}")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        linea = linecache.getline(ruta_metadata, int(idx) + 1)
        obj = json.loads(linea)
        print(f"  #{rank} score={score:.4f} chunk_id={obj['chunk_id']} "
              f"doc_id={obj['doc_id']} texto[:100]={obj['texto'][:100]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, help="Ruta a metadata.jsonl de entrada")
    ap.add_argument("--salida", required=True, help="Carpeta encoder_<nombre>/ de salida")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                     help="Procesar solo los primeros N chunks (prueba de concepto)")
    ap.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    ap.add_argument("--device", default=None,
                     help="cpu | cuda | cuda:0 ... (por defecto: autodetección; "
                          "usa GPU automáticamente si sentence-transformers la encuentra)")
    ap.add_argument("--no-reanudar", action="store_true",
                     help="Ignora cualquier checkpoint previo y empieza desde cero")
    ap.add_argument("--consulta-prueba", default=None,
                     help="Si se pasa, ejecuta una consulta de prueba al terminar")
    args = ap.parse_args()

    construir_indice(
        args.metadata, args.salida, batch_size=args.batch_size,
        limite=args.limit, modelo_nombre=args.modelo,
        reanudar=not args.no_reanudar, device=args.device,
    )

    if args.consulta_prueba:
        prueba_consulta(args.salida, args.consulta_prueba, modelo_nombre=args.modelo)
