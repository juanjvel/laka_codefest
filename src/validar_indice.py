#!/usr/bin/env python3
"""
validar_indice.py

Valida que un índice FAISS + metadata.jsonl ya construidos cumplan lo
exigido por la Especificación Técnica CODEFEST AD ASTRA 2026 - Etapa 1:
  - Sección 1.4: el orden de metadata.jsonl debe coincidir con los ids
    internos de FAISS.
  - Sección 4.3: soporte multilingüe nativo (ES/EN/PT).
  - Sección 8.2: similitud coseno vía vectores normalizados.

Corre cuatro chequeos, de más barato a más caro:
  1. Estructural   — tipo de índice, dimensión, conteo.
  2. Normalización — norma unitaria en una muestra de vectores.
  3. Identidad     — reconstruye el vector en la posición i y lo compara
                      contra el embedding recalculado desde el texto de
                      la línea i. Es la prueba más fuerte: si algo quedó
                      desalineado (orden distinto, texto equivocado), la
                      similitud cae lejos de 1.0 aunque los conteos
                      cuadren.
  4. Humo multilingüe — una consulta por fenómeno, cada una en un idioma
                      distinto (es/en/pt), para confirmar que el encoder
                      recupera resultados cruzando idiomas.

Uso:
    python validar_indice.py --carpeta base_vectorial/encoder_multilingual-e5-small
    python validar_indice.py --carpeta ... --n-muestras-identidad 50
"""
import argparse
import json
import linecache
import random
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODELO_POR_DEFECTO = "intfloat/multilingual-e5-small"
PREFIJO_PASSAGE = "passage: "
PREFIJO_QUERY = "query: "

# Una consulta por fenómeno, en tres idiomas distintos: prueba explícita
# de la Sección 4.3 (soporte multilingüe), no solo de relevancia temática.
CONSULTAS_HUMO = [
    (1, "es", "impacto de la inteligencia artificial en la defensa nacional"),
    (2, "en", "space debris collision risk in low Earth orbit"),
    (3, "pt", "migração e dinâmicas territoriais na Amazônia"),
]


def chequeo_estructural(index, n_meta):
    print("== 1. Chequeo estructural ==")
    ok = True
    if not isinstance(index, faiss.IndexFlatIP):
        print(f"  [FALLA] tipo de índice es {type(index).__name__}, se esperaba IndexFlatIP")
        ok = False
    else:
        print("  [ok] tipo de índice: IndexFlatIP")
    if index.d != 384:
        print(f"  [FALLA] dimensión = {index.d}, se esperaba 384")
        ok = False
    else:
        print(f"  [ok] dimensión: {index.d}")
    if index.ntotal != n_meta:
        print(f"  [FALLA] index.ntotal={index.ntotal} != líneas de metadata.jsonl={n_meta}")
        ok = False
    else:
        print(f"  [ok] conteo alineado: {index.ntotal} vectores == {n_meta} líneas")
    return ok


def chequeo_normalizacion(index, n_muestras=200):
    print("== 2. Chequeo de normalización (norma unitaria) ==")
    idx_muestra = sorted(random.sample(range(index.ntotal), min(n_muestras, index.ntotal)))
    vecs = np.stack([index.reconstruct(i) for i in idx_muestra])
    normas = np.linalg.norm(vecs, axis=1)
    fuera_de_rango = int(np.sum(np.abs(normas - 1.0) > 1e-3))
    if fuera_de_rango:
        print(f"  [FALLA] {fuera_de_rango}/{len(idx_muestra)} vectores con norma != 1.0 "
              f"(min={normas.min():.4f}, max={normas.max():.4f})")
        return False
    print(f"  [ok] {len(idx_muestra)} vectores muestreados, todos con norma ~1.0 "
          f"(min={normas.min():.6f}, max={normas.max():.6f})")
    return True


def chequeo_identidad_vector_texto(index, ruta_metadata, modelo, n_muestras=20):
    """La prueba más fuerte: reconstruye el vector en la posición i,
    recalcula el embedding desde el texto de la línea i, y compara.
    Si algo se desalineó, la similitud coseno cae lejos de 1.0."""
    print(f"== 3. Chequeo de identidad vector<->texto ({n_muestras} muestras) ==")
    idx_muestra = random.sample(range(index.ntotal), min(n_muestras, index.ntotal))
    peor = 1.0
    fallas = []
    for i in idx_muestra:
        linea = linecache.getline(str(ruta_metadata), i + 1)
        obj = json.loads(linea)
        v_guardado = index.reconstruct(i)
        v_recalculado = modelo.encode(
            [PREFIJO_PASSAGE + obj["texto"]],
            normalize_embeddings=True, convert_to_numpy=True,
        )[0]
        sim = float(np.dot(v_guardado, v_recalculado))
        peor = min(peor, sim)
        if sim < 0.999:
            fallas.append((i, obj["chunk_id"], sim))
    if fallas:
        print(f"  [FALLA] {len(fallas)}/{len(idx_muestra)} posiciones no coinciden:")
        for i, cid, sim in fallas[:10]:
            print(f"     id={i} chunk_id={cid} similitud={sim:.4f}")
        return False
    print(f"  [ok] {len(idx_muestra)} posiciones verificadas, similitud mínima={peor:.6f}")
    return True


def prueba_humo_multilingue(index, ruta_metadata, modelo, k=5):
    print("== 4. Prueba de humo multilingüe (una consulta por fenómeno) ==")
    for fenomeno_esperado, idioma, consulta in CONSULTAS_HUMO:
        q = modelo.encode(
            [PREFIJO_QUERY + consulta], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, ids = index.search(q, k)
        print(f"\n  Fenómeno {fenomeno_esperado} [{idioma}]: {consulta!r}")
        coinciden = 0
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
            linea = linecache.getline(str(ruta_metadata), int(idx) + 1)
            obj = json.loads(linea)
            marca = "*" if obj["fenomeno"] == fenomeno_esperado else " "
            if obj["fenomeno"] == fenomeno_esperado:
                coinciden += 1
            snippet = obj["texto"][:90].replace("\n", " ")
            print(f"    {marca} #{rank} score={score:.3f} fenomeno={obj['fenomeno']} "
                  f"formato={obj['formato']} {snippet!r}")
        print(f"    -> {coinciden}/{k} resultados del fenómeno esperado")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carpeta", required=True,
                     help="carpeta encoder_.../ con index.faiss y metadata.jsonl")
    ap.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    ap.add_argument("--n-muestras-identidad", type=int, default=20)
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    ruta_index = carpeta / "index.faiss"
    ruta_metadata = carpeta / "metadata.jsonl"

    print(f"[info] Cargando índice: {ruta_index}")
    index = faiss.read_index(str(ruta_index))
    with open(ruta_metadata, encoding="utf-8") as f:
        n_meta = sum(1 for _ in f)

    ok_estructural = chequeo_estructural(index, n_meta)
    ok_normalizacion = chequeo_normalizacion(index)

    print(f"[info] Cargando encoder para pruebas semánticas: {args.modelo}")
    modelo = SentenceTransformer(args.modelo)

    ok_identidad = chequeo_identidad_vector_texto(
        index, ruta_metadata, modelo, n_muestras=args.n_muestras_identidad)
    prueba_humo_multilingue(index, ruta_metadata, modelo)

    print("\n" + "=" * 60)
    if ok_estructural and ok_normalizacion and ok_identidad:
        print("RESULTADO: el índice pasa todos los chequeos automáticos.")
    else:
        print("RESULTADO: hay fallas -- revisa los mensajes [FALLA] arriba.")


if __name__ == "__main__":
    main()
