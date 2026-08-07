"""
Pipeline de limpieza y extracción — CODEFEST AD ASTRA 2026
Alcance: Fenómeno 2 (Seguridad y Entorno Espacial) únicamente.
Formatos cubiertos: PDF (con OCR de respaldo), JSON (dinámico), CSV (multidelimitador, TSV), Imagen (OCR), Texto.
"""

import os
import io
import json
import re
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
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False
    print("ADVERTENCIA: pytesseract no está instalado. OCR desactivado (PDFs escaneados e imágenes).")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_DISPONIBLE = True
except ImportError:
    PDF2IMAGE_DISPONIBLE = False
    print("ADVERTENCIA: pdf2image no está instalado. OCR de respaldo para PDFs escaneados desactivado.")

try:
    from PIL import Image
    PIL_DISPONIBLE = True
except ImportError:
    PIL_DISPONIBLE = False
    print("ADVERTENCIA: Pillow no está instalado. Extracción de imágenes (Tipo=Imagen/Otro) desactivada.")

try:
    import pillow_avif  # noqa: F401 — registra el decodificador AVIF en Pillow.
except ImportError:
    pass  # Solo relevante para el único archivo .avif de F2; si falta, ese documento cae a "pendientes".

FENOMENO_OBJETIVO = "F2"


class CodefestTextCleaner:
    @staticmethod
    def reparar_guion_fin_linea(texto):
        """Repara palabras cortadas por guion de fin de línea (ej. 'coopera-\\nción' -> 'cooperación').
        Debe aplicarse mientras el salto de línea original todavía esté presente — una vez que un
        extractor colapsa ese salto a un espacio (p.ej. al unir las líneas envueltas de un bloque de
        PDF), el patrón ya no es detectable."""
        return re.sub(r'-\n(?=\w)', '', texto)

    @staticmethod
    def clean_text(raw_text):
        """Normaliza espacios/tabs dentro de cada línea, pero preserva los saltos de línea recibidos
        como separadores de párrafo (uno por párrafo/registro, según el extractor de origen) en vez de
        colapsar todo el texto a una sola línea. Esto es lo que permite que las etapas de chunking
        posteriores usen párrafos como frontera natural, además de oraciones."""
        texto = CodefestTextCleaner.reparar_guion_fin_linea(raw_text)
        texto = re.sub(r'[ \t\r\f\v]+', ' ', texto)
        parrafos = [p.strip() for p in texto.split('\n') if p.strip()]
        return '\n'.join(parrafos)

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
        """Extrae texto de PDFs agrupando por bloques de layout ("blocks") en vez de por línea visual
        plana ("text"). PyMuPDF en modo "text" numera una línea por cada línea envuelta por el ancho de
        columna, sin distinguir un salto de línea real de fin de párrafo de uno de simple ajuste de texto
        — con ese modo, toda noción de párrafo se pierde antes de llegar a clean_text(). El modo "blocks"
        agrupa las líneas que pertenecen a un mismo bloque de layout (aproximación robusta a un párrafo
        en la mayoría de reportes con una o dos columnas), preservando esa frontera como un solo salto de
        línea por bloque. Si el texto está vacío (escaneado), aplica OCR como fallback."""
        full_path = os.path.join(self.base_dir, relative_path)
        paginas = []
        try:
            doc = fitz.open(full_path)
            for page in doc:
                bloques = page.get_text("blocks", sort=True)
                parrafos_pagina = []
                for bloque in bloques:
                    texto_bloque, tipo_bloque = bloque[4], bloque[6]
                    if tipo_bloque != 0:
                        continue  # 1 = bloque de imagen; solo interesa el texto (tipo 0).
                    # Reparar guiones de fin de línea *antes* de aplanar las líneas envueltas del
                    # bloque a un espacio — una vez aplanadas, el salto de línea que delataba el
                    # corte ya no existe (ver CodefestTextCleaner.reparar_guion_fin_linea).
                    texto_bloque = CodefestTextCleaner.reparar_guion_fin_linea(texto_bloque)
                    parrafo = " ".join(l.strip() for l in texto_bloque.split("\n") if l.strip())
                    if parrafo:
                        parrafos_pagina.append(parrafo)
                paginas.append("\n".join(parrafos_pagina))
            doc.close()
        except Exception as e:
            print(f"Error procesando PDF {full_path}: {e}")
            return "", {}

        paginas_limpias = CodefestTextCleaner.eliminar_cabeceras_pies(paginas)
        texto_crudo = "\n".join(paginas_limpias)

        # --- Emergencia: OCR para PDFs Escaneados ---
        if not texto_crudo.strip() and OCR_DISPONIBLE and PDF2IMAGE_DISPONIBLE:
            try:
                print(f"[{relative_path}] Texto vacío en PDF. Aplicando OCR de respaldo...")
                images = convert_from_path(full_path)
                texto_ocr = []
                for img in images:
                    texto_ocr.append(pytesseract.image_to_string(img))
                texto_crudo = "\n".join(texto_ocr)
            except Exception as e:
                print(f"Error en OCR para {full_path}: {e}")

        return texto_crudo, {}

    def extract_json(self, relative_path):
        """Extracción dinámica de objetos JSON aislando la metadata."""
        full_path = os.path.join(self.base_dir, relative_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- Búsqueda dinámica de campos de texto ---
            titulo = data.get('title', '')
            resumen = data.get('excerpt', '')

            cuerpo = data.get('body_text', '')
            if not cuerpo:
                # Si body_text no existe, probamos con body_paragraphs
                cuerpo_lista = data.get('body_paragraphs', [])
                if isinstance(cuerpo_lista, list):
                    cuerpo = "\n".join(str(p) for p in cuerpo_lista)

            # Fallback final si los observatorios usan otras llaves
            if not cuerpo.strip():
                cuerpo = data.get('texto', '') or data.get('content', '') or data.get('body', '')

            texto_crudo = f"{titulo}\n{resumen}\n{cuerpo}".strip()

            # --- Aislamiento riguroso de Metadata ---
            metadata = {
                'url': data.get('url', ''),
                'authors': data.get('authors', []),
                'date': data.get('date', ''),
                'tags': data.get('tags', []) or data.get('topics', [])
            }
            return texto_crudo, metadata
        except Exception as e:
            print(f"Error procesando JSON {full_path}: {e}")
            return "", {}

    def extract_tabular(self, relative_path, is_csv=True):
        """Extrae tabulares incluyendo soporte dual para CSVs con punto y coma y tabulaciones."""
        full_path = os.path.join(self.base_dir, relative_path)
        texto_crudo = ""
        try:
            if is_csv:
                # --- Manejo de delimitadores y codificaciones ---
                try:
                    df = pd.read_csv(full_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(full_path, encoding='latin-1')
                except Exception:
                    try:
                        # Intento con punto y coma
                        df = pd.read_csv(full_path, encoding='utf-8', sep=';')
                    except Exception:
                        # Intento con tabulación (Solución exacta para el AIINDEX)
                        df = pd.read_csv(full_path, encoding='utf-8', sep='\t')

                # Si se leyó todo como una columna gigante, forzamos separadores alternativos
                if len(df.columns) == 1:
                    try:
                        df = pd.read_csv(full_path, encoding='utf-8', sep=';')
                        if len(df.columns) == 1:
                            df = pd.read_csv(full_path, encoding='utf-8', sep='\t')
                    except Exception:
                        pass
            else:
                df = pd.read_excel(full_path)

            for _, row in df.iterrows():
                fila = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                texto_crudo += fila + "\n"
        except Exception as e:
            print(f"Error procesando tabular {full_path}: {e}")
        return texto_crudo, {}

    def extract_texto(self, relative_path):
        full_path = os.path.join(self.base_dir, relative_path)
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(), {}
        except Exception as e:
            print(f"Error procesando Texto {full_path}: {e}")
            return "", {}

    def extract_imagen(self, relative_path):
        """Aplica OCR directo sobre un archivo de imagen (JPG, PNG, AVIF, ...) para recuperar texto
        legible (infografías, gráficos, capturas). Fotografías sin texto simplemente producen texto vacío
        y el documento termina en 'pendientes' con motivo 'texto_vacio_o_error'."""
        full_path = os.path.join(self.base_dir, relative_path)
        if not (OCR_DISPONIBLE and PIL_DISPONIBLE):
            return "", {}
        try:
            with Image.open(full_path) as img:
                # pytesseract solo reconoce el formato original de algunos codecs (p.ej. AVIF, WEBP no
                # siempre) al pasarle el objeto PIL directamente; re-codificar a PNG en memoria lo evita.
                buffer = io.BytesIO()
                img.convert("RGB").save(buffer, format="PNG")
                buffer.seek(0)
                with Image.open(buffer) as img_png:
                    texto_ocr = pytesseract.image_to_string(img_png)
            return texto_ocr, {}
        except Exception as e:
            print(f"Error procesando Imagen {full_path}: {e}")
            return "", {}

    def extract_catalogo(self, relative_path, es_csv):
        """Catálogos de scraping (CSIS_catalog-2.csv/.json): manifiestos con URLs y metadata de
        descarga sobre PDFs que ni siquiera están en el corpus, no contenido documental propio. En vez
        de indexar todas las columnas de control (scraped_at, status, size_bytes, dest...) — que no
        aportan señal semántica para retrieval — se reduce cada registro a un resumen mínimo con las
        únicas columnas con algo de valor semántico: título, año, URL de la página fuente."""
        full_path = os.path.join(self.base_dir, relative_path)
        try:
            if es_csv:
                df = pd.read_csv(full_path, encoding='utf-8')
                registros = df.to_dict(orient='records')
            else:
                with open(full_path, 'r', encoding='utf-8') as f:
                    registros = json.load(f)
                if not isinstance(registros, list):
                    registros = [registros]

            lineas = []
            for reg in registros:
                titulo = reg.get('titulo', '')
                anio = reg.get('año', reg.get('anio', ''))
                url = reg.get('page_url', '')
                if titulo or url:
                    lineas.append(f"titulo: {titulo} | año: {anio} | page_url: {url}")
            return "\n".join(lineas), {}
        except Exception as e:
            print(f"Error procesando catálogo {full_path}: {e}")
            return "", {}


def ejecutar_pipeline_extraccion(excel_indice_path, base_corpus_dir, output_jsonl_path):
    df_inventario = pd.read_excel(excel_indice_path, sheet_name='Inventario de Archivos')
    df_f2 = df_inventario[df_inventario['Fenómeno'] == FENOMENO_OBJETIVO]
    print(f"Iniciando extracción para {len(df_f2)} documentos de {FENOMENO_OBJETIVO}...")

    extractor = CodefestExtractor(base_corpus_dir)
    cleaner = CodefestTextCleaner()
    base_documental = []
    pendientes = []

    # --- Catálogos de scraping: no tienen contenido documental propio, solo metadata sobre PDFs que
    # ni siquiera están en el corpus. En vez de descartarlos, se procesan con extract_catalogo() para
    # obtener un resumen mínimo (titulo/año/page_url) — mismos archivos que F1 habría ignorado, pero
    # aquí sí se indexan (a decisión explícita, ver f2-pipeline-implementation memory).
    catalogos_de_metadata = ['CSIS_catalog-2.json', 'CSIS_catalog-2.csv']

    for _, row in df_f2.iterrows():
        doc_id = str(row['DOC_ID'])
        nombre_archivo = str(row['Nombre estandarizado'])
        carpeta = str(row['Carpeta'])
        tipo = str(row['Tipo']).lower()

        relative_path = os.path.join(carpeta, nombre_archivo)
        texto_crudo = ""
        metadata_extra = {}

        if nombre_archivo in catalogos_de_metadata:
            texto_crudo, metadata_extra = extractor.extract_catalogo(relative_path, es_csv=(tipo == 'csv'))
        elif tipo == 'pdf':
            texto_crudo, metadata_extra = extractor.extract_pdf(relative_path)
        elif tipo == 'json':
            texto_crudo, metadata_extra = extractor.extract_json(relative_path)
        elif tipo == 'csv':
            texto_crudo, metadata_extra = extractor.extract_tabular(relative_path, is_csv=True)
        elif tipo in ('xlsx', 'excel'):
            texto_crudo, metadata_extra = extractor.extract_tabular(relative_path, is_csv=False)
        elif tipo == 'texto':
            texto_crudo, metadata_extra = extractor.extract_texto(relative_path)
        elif tipo in ('imagen', 'otro'):
            # En F2 el único registro con Tipo='Otro' es una imagen .avif mal clasificada en el índice;
            # no asumir lo mismo para otros fenómenos (p.ej. F3 usa 'Otro' para mapas .pbf).
            texto_crudo, metadata_extra = extractor.extract_imagen(relative_path)
        else:
            pendientes.append({"doc_id": doc_id, "tipo": tipo, "motivo": "tipo_no_manejado"})
            continue

        if texto_crudo.strip():
            texto_limpio = cleaner.clean_text(texto_crudo)
            idioma_detectado = cleaner.detectar_idioma(texto_limpio)

            documento_procesado = {
                "doc_id": doc_id,
                "fuente": nombre_archivo,
                "formato": tipo,
                "fenomeno": FENOMENO_OBJETIVO,
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

    _validar_conteo_total(excel_indice_path, len(base_documental), len(pendientes))

    return base_documental, pendientes


def _validar_conteo_total(excel_indice_path, n_procesados, n_pendientes):
    df_resumen = pd.read_excel(excel_indice_path, sheet_name='Resumen por Fenomeno')
    fila = df_resumen[df_resumen['Fenómeno'].str.startswith(FENOMENO_OBJETIVO, na=False)]

    if fila.empty:
        return

    total_esperado = int(fila.iloc[0]['Total datos'])
    total_obtenido = n_procesados + n_pendientes
    print(f"Validación — {FENOMENO_OBJETIVO}: Esperado={total_esperado} | Procesados (incluyendo catálogos)+Pendientes={total_obtenido}")


# Bloque de ejecución principal
docs_limpios, pendientes = ejecutar_pipeline_extraccion(
      excel_indice_path="../data/raw/Indice_Datos_Codefest.xlsx",
      base_corpus_dir="../data/raw/",
      output_jsonl_path="../data/clean/f2_documentos.jsonl" )
