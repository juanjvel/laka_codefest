"""
Pipeline de limpieza y extracción — CODEFEST AD ASTRA 2026
Alcance: Fenómeno 3 (Dinámicas Territoriales) únicamente.
Formatos cubiertos: PDF (con OCR de respaldo), JSON (dinámico), CSV (multidelimitador),
PBF/Mapbox Vector Tiles, Texto (respaldo).

NOTA IMPORTANTE: el .pbf de este fenómeno NO es OSM PBF (osmium/pyrosm), es Mapbox
Vector Tile (MVT) — estructura de teselas tiles/{z}/{x}/{y}.pbf. Se decodifica con
la librería `mapbox-vector-tile`, no con herramientas de OpenStreetMap.
"""

import os
import re
import json
import pandas as pd
import fitz  # PyMuPDF

# --- Dependencias Opcionales ---
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_DISPONIBLE = True
except ImportError:
    LANGDETECT_DISPONIBLE = False
    print("ADVERTENCIA: langdetect no está instalado.")

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False
    print("ADVERTENCIA: pytesseract o pdf2image no están instalados. OCR de respaldo desactivado.")

try:
    import mapbox_vector_tile
    MVT_DISPONIBLE = True
except ImportError:
    MVT_DISPONIBLE = False
    print("ADVERTENCIA: mapbox-vector-tile no está instalado. pip install mapbox-vector-tile --break-system-packages")

FENOMENO_OBJETIVO = "F3"

# Patrón para detectar archivos que probablemente son catálogos/índices de metadata
# sin texto sustantivo (más variado que en F1: catalog-2.json, mapp-catalog.csv,
# tiles-index.json, etc.). VERIFICAR EN LA POC antes de confiar ciegamente en esto.
PATRON_CATALOGO = re.compile(r'(catalog|tiles-index)', re.IGNORECASE)


class CodefestTextCleaner:
    @staticmethod
    def clean_text(raw_text):
        """Normaliza espacios/tabs dentro de cada línea y repara palabras cortadas por guion de fin
        de línea, pero preserva los saltos de línea como separadores de fila/feature en vez de
        colapsar todo a una sola línea (antes: re.sub(r'\\s+', ' ', texto) destruía la frontera
        entre filas de CSV y entre features de un tile PBF/MVT concatenados, produciendo un blob
        sin puntuación real que en chunking se partía a ciegas cada 250 palabras). El chunking
        ahora trata csv como formato tabular y separa por línea directamente, así que esta
        preservación es la que lo hace posible."""
        texto = re.sub(r'-\n(?=\w)', '', raw_text)
        texto = re.sub(r'[ \t\r\f\v]+', ' ', texto)
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        return '\n'.join(lineas)

    @staticmethod
    def eliminar_cabeceras_pies(paginas_texto, min_repeticiones=3):
        if len(paginas_texto) < min_repeticiones:
            return paginas_texto

        conteo_lineas = {}
        for pagina in paginas_texto:
            lineas = {l.strip() for l in pagina.split("\n") if l.strip()}
            for linea in lineas:
                if 0 < len(linea) < 120:
                    conteo_lineas[linea] = conteo_lineas.get(linea, 0) + 1

        lineas_repetidas = {l for l, c in conteo_lineas.items() if c >= min_repeticiones}

        paginas_limpias = []
        for pagina in paginas_texto:
            lineas_filtradas = [l for l in pagina.split("\n") if l.strip() not in lineas_repetidas]
            paginas_limpias.append("\n".join(lineas_filtradas))
        return paginas_limpias

    @staticmethod
    def detectar_idioma(texto):
        if not LANGDETECT_DISPONIBLE or not texto.strip():
            return "desconocido"
        try:
            return detect(texto[:1000])
        except LangDetectException:
            return "desconocido"


class CodefestExtractor:
    def __init__(self, base_corpus_dir):
        self.base_dir = base_corpus_dir

    def extract_pdf(self, relative_path):
        """Extrae texto de PDFs. Si el texto está vacío (escaneado), aplica OCR como fallback."""
        full_path = os.path.join(self.base_dir, relative_path)
        paginas = []
        try:
            doc = fitz.open(full_path)
            for page in doc:
                paginas.append(page.get_text("text"))
            doc.close()
        except Exception as e:
            print(f"Error procesando PDF {full_path}: {e}")
            return "", {}

        paginas_limpias = CodefestTextCleaner.eliminar_cabeceras_pies(paginas)
        texto_crudo = "\n".join(paginas_limpias)

        if not texto_crudo.strip() and OCR_DISPONIBLE:
            try:
                print(f"[{relative_path}] Texto vacío en PDF. Aplicando OCR de respaldo...")
                images = convert_from_path(full_path)
                texto_ocr = []
                for img in images:
                    texto_ocr.append(pytesseract.image_to_string(img, lang='spa+eng'))
                texto_crudo = "\n".join(texto_ocr)
            except Exception as e:
                print(f"Error en OCR para {full_path}: {e}")

        return texto_crudo, {}

    def extract_json(self, relative_path):
        """Extracción dinámica de objetos JSON aislando la metadata (8 observatorios con esquemas distintos)."""
        full_path = os.path.join(self.base_dir, relative_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            titulo = data.get('title', '')
            resumen = data.get('excerpt', '')

            cuerpo = data.get('body_text', '')
            if not cuerpo:
                cuerpo_lista = data.get('body_paragraphs', [])
                if isinstance(cuerpo_lista, list):
                    cuerpo = "\n".join(str(p) for p in cuerpo_lista)

            if not cuerpo.strip():
                cuerpo = data.get('texto', '') or data.get('content', '') or data.get('body', '')

            # NOTA F3: con 8 observatorios distintos es probable que aparezcan más
            # variantes de llaves que en F1. Si al correr la PoC ves muchos
            # texto_vacio_o_error en JSON, revisa las llaves reales de esos archivos
            # y añádelas aquí.
            if not cuerpo.strip():
                cuerpo = data.get('description', '') or data.get('summary', '')

            texto_crudo = f"{titulo}\n{resumen}\n{cuerpo}".strip()

            metadata = {
                'url': data.get('url', ''),
                'authors': data.get('authors', []),
                'date': data.get('date', ''),
                'tags': data.get('tags', [])
            }
            return texto_crudo, metadata
        except Exception as e:
            print(f"Error procesando JSON {full_path}: {e}")
            return "", {}

    @staticmethod
    def _leer_csv_robusto(full_path):
        """Detecta separador y encoding automáticamente en vez de asumir uno fijo."""
        for encoding in ('utf-8', 'latin-1'):
            try:
                df = pd.read_csv(full_path, encoding=encoding, sep=None, engine='python')
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(full_path, encoding='latin-1', engine='python')

    def extract_tabular(self, relative_path):
        """Extrae CSV con detección automática de separador y codificación."""
        full_path = os.path.join(self.base_dir, relative_path)
        texto_crudo = ""
        try:
            df = self._leer_csv_robusto(full_path)
            for _, row in df.iterrows():
                fila = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                texto_crudo += fila + "\n"
        except Exception as e:
            print(f"Error procesando tabular {full_path}: {e}")
        return texto_crudo, {}

    def extract_pbf(self, relative_path):
        """
        Decodifica teselas Mapbox Vector Tile (MVT), NO OSM PBF.
        Estructura esperada: .../tiles/{z}/{x}/{y}.pbf
        Extrae nombres de capa y propiedades de cada feature como texto,
        y conserva z/x/y como metadata espacial.
        """
        if not MVT_DISPONIBLE:
            return "", {}

        full_path = os.path.join(self.base_dir, relative_path)
        try:
            with open(full_path, 'rb') as f:
                tile_bytes = f.read()
            decoded = mapbox_vector_tile.decode(tile_bytes)
        except Exception as e:
            print(f"Error procesando PBF/MVT {full_path}: {e}")
            return "", {}

        # z/x/y vienen codificados en la ruta: tiles/{z}/{x}/{y}.pbf
        partes = relative_path.replace("\\", "/").split("/")
        tile_z, tile_x, tile_y = (None, None, None)
        if len(partes) >= 3:
            tile_y = os.path.splitext(partes[-1])[0]
            tile_x = partes[-2]
            tile_z = partes[-3]

        fragmentos = []
        for nombre_capa, contenido in decoded.items():
            features = contenido.get('features', [])
            if not features:
                continue
            fragmentos.append(f"Capa: {nombre_capa} ({len(features)} elementos)")
            for feature in features:
                props = feature.get('properties', {})
                if props:
                    fila = " | ".join(f"{k}: {v}" for k, v in props.items())
                    fragmentos.append(fila)

        texto_crudo = "\n".join(fragmentos)
        metadata = {"tile_z": tile_z, "tile_x": tile_x, "tile_y": tile_y}
        return texto_crudo, metadata

    def extract_texto(self, relative_path):
        full_path = os.path.join(self.base_dir, relative_path)
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(), {}
        except Exception as e:
            print(f"Error procesando Texto {full_path}: {e}")
            return "", {}


def ejecutar_pipeline_extraccion(excel_indice_path, base_corpus_dir, output_jsonl_path):
    df_inventario = pd.read_excel(excel_indice_path, sheet_name='Inventario de Archivos')
    df_f3 = df_inventario[df_inventario['Fenómeno'] == FENOMENO_OBJETIVO]
    print(f"Iniciando extracción para {len(df_f3)} documentos de {FENOMENO_OBJETIVO}...")

    extractor = CodefestExtractor(base_corpus_dir)
    cleaner = CodefestTextCleaner()
    base_documental = []
    pendientes = []
    catalogos_ignorados_reales = []  # se llena con lo que realmente se encuentra

    for _, row in df_f3.iterrows():
        doc_id = str(row['DOC_ID'])
        nombre_archivo = str(row['Nombre estandarizado'])
        carpeta = str(row['Carpeta'])
        tipo = str(row['Tipo']).lower()

        if PATRON_CATALOGO.search(nombre_archivo):
            catalogos_ignorados_reales.append(doc_id)
            print(f"Ignorando posible catálogo de metadatos: {nombre_archivo}")
            continue

        relative_path = os.path.join(carpeta, nombre_archivo)
        texto_crudo = ""
        metadata_extra = {}

        formato_real = tipo
        if tipo == 'pdf':
            texto_crudo, metadata_extra = extractor.extract_pdf(relative_path)
        elif tipo == 'json':
            texto_crudo, metadata_extra = extractor.extract_json(relative_path)
        elif tipo == 'csv':
            texto_crudo, metadata_extra = extractor.extract_tabular(relative_path)
        elif tipo == 'otro':
            # En F3, 'Otro' son exclusivamente los .pbf (Mapbox Vector Tiles).
            # Se guarda formato='pbf' (no el 'otro' generico del inventario)
            # porque su texto es "una linea por feature", con la misma
            # estructura de fila que un CSV -- chunking_semantico.py
            # necesita saber esto para tratarlo como tabular y no como
            # prosa (mismo problema que tenia el CSV del AI Index).
            formato_real = 'pbf'
            texto_crudo, metadata_extra = extractor.extract_pbf(relative_path)
        elif tipo == 'texto':
            texto_crudo, metadata_extra = extractor.extract_texto(relative_path)
        else:
            pendientes.append({"doc_id": doc_id, "tipo": tipo, "motivo": "tipo_no_manejado"})
            continue

        if texto_crudo.strip():
            texto_limpio = cleaner.clean_text(texto_crudo)
            idioma_detectado = cleaner.detectar_idioma(texto_limpio)

            documento_procesado = {
                "doc_id": doc_id,
                "fuente": nombre_archivo,
                "formato": formato_real,
                # Tabla 1 del spec exige "fenomeno" como entero (1, 2 o 3).
                "fenomeno": int(FENOMENO_OBJETIVO[1:]),
                "texto_limpio": texto_limpio,
                "idioma_detectado": idioma_detectado,
                **metadata_extra
            }
            base_documental.append(documento_procesado)
        else:
            pendientes.append({"doc_id": doc_id, "tipo": tipo, "motivo": "texto_vacio_o_error"})

    print(f"Extracción finalizada. Documentos listos para chunking: {len(base_documental)}")

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for doc in base_documental:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    if pendientes:
        print(f"ATENCIÓN: {len(pendientes)} documentos no se procesaron.")
        pendientes_path = output_jsonl_path.replace(".jsonl", "_pendientes.jsonl")
        with open(pendientes_path, 'w', encoding='utf-8') as f:
            for p in pendientes:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    _validar_conteo_total(
        excel_indice_path,
        len(base_documental) + len(catalogos_ignorados_reales),
        len(pendientes)
    )

    return base_documental, pendientes


def _validar_conteo_total(excel_indice_path, n_procesados, n_pendientes):
    df_resumen = pd.read_excel(excel_indice_path, sheet_name='Resumen por Fenomeno')
    fila = df_resumen[df_resumen['Fenómeno'].str.startswith(FENOMENO_OBJETIVO, na=False)]

    if fila.empty:
        return

    total_esperado = int(fila.iloc[0]['Total datos'])
    total_obtenido = n_procesados + n_pendientes
    print(f"Validación — {FENOMENO_OBJETIVO}: Esperado={total_esperado} | Procesados (incl. catálogos)+Pendientes={total_obtenido}")


# Bloque de ejecución principal
if __name__ == "__main__":
    # NOTA: confirma que Indice_Datos_Codefest.xlsx vive realmente en data/raw/
    # antes de correr esto.
    docs_limpios, pendientes = ejecutar_pipeline_extraccion(
        excel_indice_path="../data/raw/Indice_Datos_Codefest.xlsx",
        base_corpus_dir="../data/raw/",
        output_jsonl_path="../data/clean/f3_documentos.jsonl"
    )
