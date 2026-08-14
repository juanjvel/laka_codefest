#!/usr/bin/env python3
"""
evaluar_resultados.py

Mide localmente la calidad de un `resultados.jsonl` ya generado, con las
metricas de la Seccion 10.2 del spec: NDCG@10 sobre fragmentos y F1@3 sobre
documentos.

Es una herramienta de desarrollo, no un entregable: sirve para afinar los
parametros de `entrega/generador.py` con evidencia en vez de a ojo.

Las funciones de carga del ground truth y de metricas son una copia adaptada de
`benchmark_modelos.py`. Se copian en vez de importarse porque ese modulo hace
`nltk.download('punkt')` al importarse, ademas de cargar torch, faiss y
sentence-transformers; en una maquina sin `nltk_data` ni acceso a los
servidores de NLTK, importarlo cuelga indefinidamente. Este script solo
necesita pandas, numpy y openpyxl, y asi corre en segundos.

ADVERTENCIA sobre la interpretacion de los numeros: el ground truth real de la
competencia no es publico. Esto mide contra la anotacion manual del equipo, y
la relevancia de un fragmento se decide por solapamiento de texto, que es una
aproximacion del criterio descrito en la Seccion 10.2.1. Sirve para comparar
configuraciones entre si, no para predecir la puntuacion final absoluta.

Uso:
    python evaluar_resultados.py --detalle
"""
import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RUTA_RESULTADOS = BASE_DIR.parent / "entrega" / "resultados.jsonl"
RUTA_GROUND_TRUTH = BASE_DIR.parent / "data" / "raw" / "FASE ORDENADA CODEFEST.xlsx"
RUTA_INDICE_MAESTRO = BASE_DIR.parent / "data" / "raw" / "Indice_Datos_Codefest.xlsx"
RUTA_CONSULTAS = BASE_DIR.parent / "data" / "raw" / "Extracto_Preguntas_50_v2.pdf"

# Umbral para emparejar la pregunta del Excel con la del archivo oficial de
# consultas. El texto anotado a mano difiere en tildes, espacios o algun
# conector, asi que la igualdad exacta pierde emparejamientos validos.
UMBRAL_EMPAREJAR_PREGUNTA = 0.90

# Umbral de solapamiento para considerar que un fragmento entregado corresponde
# al excerto anotado a mano en la columna FRAGMENTO.
UMBRAL_COINCIDENCIA_FRAGMENTO = 0.5


# ---------------------------------------------------------------------------
# Carga del ground truth (adaptado de benchmark_modelos.py)
# ---------------------------------------------------------------------------

def normalizar_nombre(nombre: str) -> str:
    """Normaliza un nombre de archivo para compararlo entre esquemas distintos
    (el DOCUMENTO del ground truth vs. el Nombre estandarizado del indice)."""
    nombre = str(nombre).strip().lower()
    nombre = re.sub(r"\.(pdf|json|csv|xlsx)$", "", nombre)
    nombre = re.sub(r"[^a-z0-9]+", "-", nombre)
    return nombre.strip("-")


def construir_crosswalk_doc_id(ruta_indice_maestro: Path):
    """Mapa nombre_de_archivo -> doc_id canonico, desde la hoja 'Inventario de
    Archivos'. Necesario porque el DOCUMENTO del ground truth usa el nombre del
    archivo y un id ad-hoc del anotador ('DOC-0296'), no el doc_id del pipeline
    ('F3-MAPPOEA-020')."""
    import openpyxl
    wb = openpyxl.load_workbook(ruta_indice_maestro, data_only=True, read_only=True)
    ws = wb["Inventario de Archivos"]
    sin_prefijo, completo = {}, {}
    for fila in ws.iter_rows(min_row=2, values_only=True):
        _fenomeno, _observatorio, codigo, doc_id, nombre_std, _carpeta, _tipo = fila
        if not nombre_std or not doc_id:
            continue
        completo[normalizar_nombre(nombre_std)] = doc_id
        if codigo:
            recortado = re.sub(rf"^{re.escape(str(codigo))}_", "", str(nombre_std),
                               flags=re.IGNORECASE)
            sin_prefijo[normalizar_nombre(recortado)] = doc_id
    return sin_prefijo, completo


def resolver_doc_id(celda, sin_prefijo: dict, completo: dict):
    """La celda DOCUMENTO trae el nombre de archivo y el id del anotador
    separados por un salto de linea; solo el nombre de archivo es fiable."""
    if pd.isna(celda):
        return None
    nombre = str(celda).split("\n")[0].strip()
    clave = normalizar_nombre(nombre)
    return sin_prefijo.get(clave) or completo.get(clave)


def detectar_fila_header(ruta_excel: Path, hoja: str, max_filas: int = 10) -> int:
    """Las hojas no comparten la fila de encabezado (F1 la tiene en la fila 1,
    F3 en la 0), asi que se localiza buscando la celda 'PREGUNTA'."""
    crudo = pd.read_excel(ruta_excel, sheet_name=hoja, header=None, nrows=max_filas)
    for i, fila in crudo.iterrows():
        if "PREGUNTA" in [str(v).strip().upper() for v in fila if pd.notna(v)]:
            return i
    return 0


def cargar_ground_truth(ruta_excel: Path, ruta_indice_maestro: Path) -> dict:
    sin_prefijo, completo = construir_crosswalk_doc_id(ruta_indice_maestro)
    xls = pd.ExcelFile(ruta_excel)
    ground_truth = {}
    sin_resolver = 0

    for hoja in xls.sheet_names:
        fila_header = detectar_fila_header(ruta_excel, hoja)
        df = pd.read_excel(xls, sheet_name=hoja, header=fila_header)
        if "PREGUNTA" not in df.columns:
            print(f"   [aviso] hoja {hoja!r} sin columna PREGUNTA, se omite")
            continue

        # PREGUNTA viene como celda combinada que cubre varias filas de
        # evidencia; sin el ffill esas filas de continuacion se pierden.
        df["PREGUNTA"] = df["PREGUNTA"].ffill()
        agrupado = df.dropna(subset=["PREGUNTA"]).groupby("PREGUNTA").agg({
            "FRAGMENTO": lambda x: list(x.dropna().astype(str).unique()),
            "DOCUMENTO": lambda x: list(x.dropna().astype(str).unique()),
        }).reset_index()

        for _, fila in agrupado.iterrows():
            doc_ids = []
            for celda in fila["DOCUMENTO"]:
                resuelto = resolver_doc_id(celda, sin_prefijo, completo)
                if resuelto:
                    doc_ids.append(resuelto)
                else:
                    sin_resolver += 1
            ground_truth[str(fila["PREGUNTA"]).strip()] = {
                "fragmentos_relevantes": fila["FRAGMENTO"],
                "doc_ids_relevantes": doc_ids,
            }
        print(f"   [ok] hoja {hoja!r}: header en fila {fila_header}, "
              f"{len(agrupado)} preguntas")

    if sin_resolver:
        print(f"   [aviso] {sin_resolver} referencias DOCUMENTO sin resolver a doc_id")
    return ground_truth


# ---------------------------------------------------------------------------
# Metricas oficiales (Seccion 10.2)
# ---------------------------------------------------------------------------

def _texto_normalizado(texto) -> str:
    return re.sub(r"\s+", " ", str(texto).lower()).strip()


def fragmento_coincide(fragmento_gt: str, texto_entregado: str,
                       umbral: float = UMBRAL_COINCIDENCIA_FRAGMENTO) -> bool:
    """El ground truth marca relevancia con un excerto de texto libre, no con
    nuestro chunk_id, asi que la unica forma de saber si un fragmento entregado
    corresponde es comparar contenido (Seccion 10.2.1)."""
    a, b = _texto_normalizado(fragmento_gt), _texto_normalizado(texto_entregado)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= umbral


def calcular_ndcg_10(textos_entregados: list, fragmentos_gt: list) -> float:
    """NDCG@10 con relevancia binaria y emparejamiento uno-a-uno.

    La version de benchmark_modelos.py acredita CADA posicion del top-10 que
    coincida con algun excerto del ground truth, mientras que el IDCG asume
    como maximo len(fragmentos_gt) posiciones relevantes. Con un unico excerto
    anotado y varios fragmentos nuestros parecidos entre si, el DCG supera al
    IDCG y devuelve valores >1 (observado: 1.4307 en q004), lo que es imposible
    para un NDCG.

    Aqui cada excerto del ground truth solo puede acreditarse una vez, de forma
    que DCG <= IDCG siempre y el resultado queda en [0, 1].
    """
    fragmentos_gt = [f for f in fragmentos_gt if _texto_normalizado(f)]
    if not fragmentos_gt:
        return 0.0

    disponibles = list(range(len(fragmentos_gt)))
    dcg = 0.0
    for i, texto in enumerate(textos_entregados[:10]):
        for pos in disponibles:
            if fragmento_coincide(fragmentos_gt[pos], texto):
                dcg += 1.0 / np.log2(i + 2)
                disponibles.remove(pos)
                break

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(fragmentos_gt), 10)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def calcular_f1_3(doc_ids_entregados: list, doc_ids_gt: list) -> float:
    relevantes = set(doc_ids_gt)
    if not relevantes:
        return 0.0
    top3 = []
    for d in doc_ids_entregados:
        if d not in top3:
            top3.append(d)
        if len(top3) == 3:
            break
    interseccion = len(set(top3) & relevantes)
    if interseccion == 0:
        return 0.0
    precision = interseccion / 3.0
    recall = interseccion / min(len(relevantes), 3.0)
    return float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------
# Emparejamiento consulta <-> ground truth
# ---------------------------------------------------------------------------

def normalizar_pregunta(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", texto)).strip()


def emparejar_consulta_con_gt(texto_consulta: str, ground_truth: dict):
    objetivo = normalizar_pregunta(texto_consulta)
    if not objetivo:
        return None
    mejor, mejor_ratio = None, 0.0
    for pregunta, datos in ground_truth.items():
        candidata = normalizar_pregunta(pregunta)
        if candidata == objetivo:
            return datos
        ratio = SequenceMatcher(None, objetivo, candidata).ratio()
        if ratio > mejor_ratio:
            mejor, mejor_ratio = datos, ratio
    return mejor if mejor_ratio >= UMBRAL_EMPAREJAR_PREGUNTA else None


def cargar_consultas_oficiales(ruta: Path) -> list:
    """Reusa el parser de entrega/generador.py para no mantener una segunda
    copia de las 50 preguntas. Ese modulo solo importa la libreria estandar a
    nivel de modulo (faiss/sentence-transformers son imports diferidos), asi
    que importarlo es barato."""
    sys.path.insert(0, str(BASE_DIR.parent / "entrega"))
    from generador import cargar_consultas, validar_consultas
    consultas = cargar_consultas(ruta)
    validar_consultas(consultas)
    return consultas


# ---------------------------------------------------------------------------

def evaluar(ruta_resultados: Path, ruta_gt: Path, ruta_indice: Path,
            ruta_consultas: Path, detalle: bool) -> None:
    for ruta in (ruta_resultados, ruta_gt, ruta_indice):
        if not ruta.exists():
            sys.exit(f"[ERROR] No se encontro {ruta}")

    resultados = {}
    with open(ruta_resultados, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                obj = json.loads(linea)
                resultados[obj["query_id"]] = obj

    consultas = cargar_consultas_oficiales(ruta_consultas)
    print(f"[info] Ground truth: {ruta_gt.name}")
    ground_truth = cargar_ground_truth(ruta_gt, ruta_indice)
    if not ground_truth:
        sys.exit("[ERROR] El ground truth se cargo vacio.")

    filas, sin_gt = [], []
    for consulta in consultas:
        query_id = consulta["query_id"]
        resultado = resultados.get(query_id)
        if resultado is None:
            sys.exit(f"[ERROR] Falta {query_id} en {ruta_resultados}")

        datos_gt = emparejar_consulta_con_gt(consulta["texto_query"], ground_truth)
        if datos_gt is None:
            sin_gt.append(query_id)
            continue

        ndcg = calcular_ndcg_10([f["text"] for f in resultado["fragments"]],
                                datos_gt["fragmentos_relevantes"])
        f1 = calcular_f1_3([d["doc_id"] for d in resultado["documents"]],
                           datos_gt["doc_ids_relevantes"])
        filas.append({
            "query_id": query_id, "ndcg": ndcg, "f1": f1,
            "n_frag_gt": len(datos_gt["fragmentos_relevantes"]),
            "n_docs_gt": len(set(datos_gt["doc_ids_relevantes"])),
        })

    if not filas:
        sys.exit("[ERROR] Ninguna consulta pudo emparejarse con el ground truth.")

    ndcg_medio = float(np.mean([f["ndcg"] for f in filas]))
    f1_medio = float(np.mean([f["f1"] for f in filas]))

    print("\n" + "=" * 62)
    print(f"Consultas con ground truth: {len(filas)}/{len(consultas)}")
    print("-" * 62)
    print(f"  NDCG@10 (fragmentos) : {ndcg_medio:.4f}")
    print(f"  F1@3    (documentos) : {f1_medio:.4f}")
    print("=" * 62)
    if len(filas) < 20:
        print(f"AVISO: solo {len(filas)} consultas anotadas. La media es muy sensible\n"
              f"a cada una; usala para comparar configuraciones, no como estimacion.")

    if detalle:
        print(f"\n{'consulta':10s} {'NDCG@10':>9s} {'F1@3':>8s} {'frag GT':>8s} {'docs GT':>8s}")
        print("-" * 48)
        for fila in filas:
            print(f"{fila['query_id']:10s} {fila['ndcg']:9.4f} {fila['f1']:8.4f} "
                  f"{fila['n_frag_gt']:8d} {fila['n_docs_gt']:8d}")


def main():
    ap = argparse.ArgumentParser(description="Mide NDCG@10 y F1@3 de un resultados.jsonl")
    ap.add_argument("--resultados", default=str(RUTA_RESULTADOS))
    ap.add_argument("--ground-truth", default=str(RUTA_GROUND_TRUTH))
    ap.add_argument("--indice-maestro", default=str(RUTA_INDICE_MAESTRO))
    ap.add_argument("--consultas", default=str(RUTA_CONSULTAS))
    ap.add_argument("--detalle", action="store_true", help="Desglose por consulta")
    args = ap.parse_args()
    evaluar(Path(args.resultados), Path(args.ground_truth),
            Path(args.indice_maestro), Path(args.consultas), args.detalle)


if __name__ == "__main__":
    main()
