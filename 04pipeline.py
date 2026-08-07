from __future__ import annotations
import json
import sys
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
RAW_ROOT = BASE_DIR
PROCESSED_ROOT = BASE_DIR / "data" / "processed" / "F3B"
DOCUMENTOS_OUTPUT = PROCESSED_ROOT / "f3b_documentos.jsonl"
INVENTARIO_OUTPUT = PROCESSED_ROOT / "inventario.jsonl"
ERRORES_OUTPUT = PROCESSED_ROOT / "pendientes_errores.jsonl"
EXCLUIDOS_OUTPUT = PROCESSED_ROOT / "excluidos.jsonl"
VALIDACION_OUTPUT = PROCESSED_ROOT / "validacion.json"

def cargar_modulo(nombre: str, archivo: str):
    ruta = BASE_DIR / archivo
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo requerido: {ruta}")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo: {ruta}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo

inventory = cargar_modulo("inventory_f3b", "01inventory.py")
extract = cargar_modulo("extract_f3b", "02extract.py")
clean = cargar_modulo("clean_f3b", "03clean.py")

Documento = inventory.Documento
Excluido = inventory.Excluido
iter_documentos = inventory.iter_documentos
iter_excluidos = inventory.iter_excluidos
extraer = extract.extraer
ExtractionError = extract.ExtractionError
clean_record = clean.clean_record

def escribir_jsonl(path: Path, registros: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as archivo:
        for registro in registros:
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

def process_document(documento: Documento) -> dict[str, Any]:
    extracted = extraer(documento)
    cleaned = clean_record(extracted)
    texto = cleaned.pop("texto", "") or ""
    if not texto.strip():
        raise ExtractionError("El documento no produjo texto después de extracción y limpieza.")
    record = {"doc_id": documento.doc_id, "fuente": documento.fuente, "formato": documento.formato, "fenomeno": documento.fenomeno, "subfenomeno": documento.subfenomeno, "idioma": cleaned.pop("idioma", None), "texto": texto}
    record.update(cleaned)
    return record

def validar(documentos: list[Documento], procesados: list[dict[str, Any]], errores: list[dict[str, Any]], excluidos: list[Excluido]) -> dict[str, Any]:
    total_inventario = len(documentos)
    total_procesados = len(procesados)
    total_errores = len(errores)
    total_excluidos = len(excluidos)

    ids_inventario = {documento.doc_id for documento in documentos}
    ids_procesados = {registro["doc_id"] for registro in procesados}
    ids_errores = {error["doc_id"] for error in errores}

    lista_ids_procesados = [registro["doc_id"] for registro in procesados]
    duplicados_procesados = len(lista_ids_procesados) != len(set(lista_ids_procesados))

    encontrados = ids_procesados | ids_errores
    faltantes = sorted(ids_inventario - encontrados)

    procesados_fuera_inventario = sorted(ids_procesados - ids_inventario)
    errores_fuera_inventario = sorted(ids_errores - ids_inventario)

    conteo_correcto = total_inventario == total_procesados + total_errores
    validacion_ok = conteo_correcto and not faltantes and not procesados_fuera_inventario and not errores_fuera_inventario and not duplicados_procesados

    formatos_inventario = Counter(documento.formato for documento in documentos)
    formatos_procesados = Counter(registro["formato"] for registro in procesados)
    formatos_errores = Counter(error["formato"] for error in errores)
    motivos_excluidos = Counter(excluido.motivo for excluido in excluidos)

    return {
        "validacion_ok": validacion_ok,
        "conteos": {"inventariados": total_inventario, "procesados": total_procesados, "errores": total_errores, "excluidos": total_excluidos, "procesados_mas_errores": total_procesados + total_errores, "total_archivos_en_carpeta": total_inventario + total_excluidos},
        "conteo_correcto": conteo_correcto,
        "faltantes": faltantes,
        "procesados_fuera_inventario": procesados_fuera_inventario,
        "errores_fuera_inventario": errores_fuera_inventario,
        "duplicados_procesados": duplicados_procesados,
        "formatos": {"inventario": dict(formatos_inventario), "procesados": dict(formatos_procesados), "errores": dict(formatos_errores)},
        "exclusiones": {"motivos": dict(motivos_excluidos)},
    }

def run() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"No existe la carpeta de fuentes:\n{RAW_ROOT}\n\nLa estructura esperada es:\n{BASE_DIR}\\\n├── 01inventory.py\n├── 02extract.py\n├── 03clean.py\n├── 04pipeline.py\n└── F3_Dinamicas_Territoriales\\")

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    documentos = list(iter_documentos(RAW_ROOT))
    inventario = []
    for documento in documentos:
        inventario.append({"doc_id": documento.doc_id, "fuente": documento.fuente, "formato": documento.formato, "fenomeno": documento.fenomeno, "subfenomeno": documento.subfenomeno})
    escribir_jsonl(INVENTARIO_OUTPUT, inventario)

    excluidos = list(iter_excluidos(RAW_ROOT))
    registro_excluidos = []
    for excluido in excluidos:
        registro_excluidos.append({"fuente": excluido.fuente, "motivo": excluido.motivo, "fenomeno": excluido.fenomeno, "subfenomeno": excluido.subfenomeno})
    escribir_jsonl(EXCLUIDOS_OUTPUT, registro_excluidos)

    procesados = []
    errores = []
    for documento in tqdm(documentos, desc="F3-B"):
        try:
            registro = process_document(documento)
            procesados.append(registro)
        except Exception as exc:
            errores.append({"doc_id": documento.doc_id, "fuente": documento.fuente, "formato": documento.formato, "fenomeno": documento.fenomeno, "subfenomeno": documento.subfenomeno, "error": str(exc), "tipo_error": type(exc).__name__})
            continue

    escribir_jsonl(DOCUMENTOS_OUTPUT, procesados)
    escribir_jsonl(ERRORES_OUTPUT, errores)

    validacion = validar(documentos, procesados, errores, excluidos)
    VALIDACION_OUTPUT.write_text(json.dumps(validacion, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("F3-B — INVENTARIO / EXTRACCIÓN / LIMPIEZA")
    print("=" * 60)
    print(f"Inventariados : {len(documentos)}")
    print(f"Excluidos     : {len(excluidos)}")
    print(f"Procesados    : {len(procesados)}")
    print(f"Errores       : {len(errores)}")
    print("Validación    : " + ("OK" if validacion["validacion_ok"] else "ERROR"))
    print()
    print(f"Inventario : {INVENTARIO_OUTPUT}")
    print(f"Excluidos  : {EXCLUIDOS_OUTPUT}")
    print(f"Documentos : {DOCUMENTOS_OUTPUT}")
    print(f"Errores    : {ERRORES_OUTPUT}")
    print(f"Validación : {VALIDACION_OUTPUT}")

if __name__ == "__main__":
    run()