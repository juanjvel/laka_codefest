import json
import re
import time
import os
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import faiss
import torch
import nltk
from sentence_transformers import SentenceTransformer

# Asegurar disponibilidad del tokenizador de oraciones de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# =====================================================================
# 1. LAS 50 CONSULTAS OFICIALES (Sección 10.1)
# =====================================================================
CONSULTAS_OFICIALES = [
    {"query_id": "q001", "texto_query": "¿Cómo está transformando la inteligencia artificial la capacidad de los Estados para prevenir, detectar y contrarrestar amenazas NBQR?"},
    {"query_id": "q002", "texto_query": "¿Cómo están empleando los sistemas no tripulados potenciados por IA para aumentar la efectividad de las operaciones militares?"},
    {"query_id": "q003", "texto_query": "¿Qué lecciones dejan los conflictos recientes sobre el empleo de inteligencia artificial en operaciones militares?"},
    {"query_id": "q004", "texto_query": "¿Qué riesgos representa la escasez de talento especializado en inteligencia artificial para el desarrollo de capacidades de defensa?"},
    {"query_id": "q005", "texto_query": "¿Qué implicaciones estratégicas tiene la dependencia de tecnologías extranjeras de IA para la autonomía y la seguridad nacional de Colombia?"},
    {"query_id": "q006", "texto_query": "¿Qué riesgos operacionales y jurídicos implica incorporar inteligencia artificial en inteligencia militar sin una doctrina, políticas y protocolos consolidados?"},
    {"query_id": "q007", "texto_query": "¿Qué desafíos plantea el empleo de sistemas autónomos o semiautónomos frente al Derecho Internacional Humanitario y la atribución de responsabilidades?"},
    {"query_id": "q008", "texto_query": "¿Cómo están empleando las fuerzas militares el análisis de inteligencia asistido por IA, el targeting inteligente y los enjambres de drones, y qué oportunidades representan para países con geografías complejas como Colombia?"},
    {"query_id": "q009", "texto_query": "¿Cómo está redefiniendo la inteligencia artificial las tácticas y conceptos de operación en los conflictos armados contemporáneos?"},
    {"query_id": "q010", "texto_query": "¿Qué capacidades basadas en IA fortalecen la detección, identificación y neutralización de drones utilizados por actores armados?"},
    {"query_id": "q011", "texto_query": "¿Cuáles son las principales barreras para incorporar inteligencia artificial en los procesos operacionales y de inteligencia de las Fuerzas Militares?"},
    {"query_id": "q012", "texto_query": "¿Cómo puede la experiencia operacional acumulada por las Fuerzas Militares convertirse en conocimiento útil para entrenar sistemas de inteligencia artificial?"},
    {"query_id": "q013", "texto_query": "¿Cuáles son las principales amenazas cibernéticas que afectan el despliegue seguro de sistemas de inteligencia artificial en infraestructuras críticas?"},
    {"query_id": "q014", "texto_query": "¿Cómo limita la disponibilidad de infraestructura de cómputo avanzado el desarrollo de capacidades nacionales de inteligencia artificial para defensa?"},
    {"query_id": "q015", "texto_query": "¿Qué riesgos estratégicos genera la dependencia de semiconductores y hardware especializado para el desarrollo de capacidades militares basadas en IA?"},
    {"query_id": "q016", "texto_query": "¿Qué ventajas estratégicas obtienen los actores que incorporan inteligencia artificial y drones de bajo costo, y cuáles son los riesgos de retrasar la transformación tecnológica en defensa?"},
    {"query_id": "q017", "texto_query": "¿Qué restricciones impone el Derecho Internacional en el Espacio en la regulación del uso de armas?"},
    {"query_id": "q018", "texto_query": "¿Qué capacidades contraespaciales representan actualmente la mayor amenaza para los sistemas satelitales?"},
    {"query_id": "q019", "texto_query": "¿Cómo se está empleando la guerra electrónica para interferir sistemas espaciales y qué incidentes recientes lo evidencian?"},
    {"query_id": "q020", "texto_query": "¿Qué incidentes recientes de spoofing han comprometido servicios satelitales y qué vulnerabilidades revelan?"},
    {"query_id": "q021", "texto_query": "¿Qué implicaciones militares tienen las maniobras de proximidad y encuentro (RPO) realizadas por satélites?"},
    {"query_id": "q022", "texto_query": "¿Qué evidencias existen sobre el desarrollo de armas de energía dirigida con potencial empleo contra activos espaciales?"},
    {"query_id": "q023", "texto_query": "¿Cuáles son las principales vulnerabilidades cibernéticas de la infraestructura satelital?"},
    {"query_id": "q024", "texto_query": "¿Qué países están desarrollando capacidades láser con potencial empleo contra sistemas espaciales?"},
    {"query_id": "q025", "texto_query": "¿Qué implicaciones tendría el despliegue de una capacidad nuclear antisatélite en órbita?"},
    {"query_id": "q026", "texto_query": "¿Cuál ha sido el impacto de las pruebas antisatélite sobre la generación de desechos orbitales?"},
    {"query_id": "q027", "texto_query": "¿Qué desafíos plantea el uso de inteligencia artificial en las operaciones espaciales?"},
    {"query_id": "q028", "texto_query": "¿Qué capacidades estratégicas ha demostrado China mediante operaciones de servicio y reabastecimiento en órbita?"},
    {"query_id": "q029", "texto_query": "¿Qué capacidades operacionales evidencian las maniobras realizadas recientemente por satélites rusos en órbita GEO?"},
    {"query_id": "q030", "texto_query": "¿Cómo han evolucionado las operaciones espaciales dinámicas dentro de la estrategia espacial estadounidense?"},
    {"query_id": "q031", "texto_query": "¿Qué cambios doctrinales han fortalecido el empleo militar de las capacidades espaciales de Corea del Norte?"},
    {"query_id": "q032", "texto_query": "¿Qué lecciones ha dejado el conflicto entre Rusia y Ucrania sobre el empleo militar del dominio espacial?"},
    {"query_id": "q033", "texto_query": "¿Cómo utilizan los grupos armados ilegales el control territorial para sustituir funciones del Estado y consolidar su influencia sobre las comunidades?"},
    {"query_id": "q034", "texto_query": "¿Qué corredores geográficos y territorios estratégicos son priorizados por los grupos armados para fortalecer su control operacional?"},
    {"query_id": "q035", "texto_query": "¿Cómo contribuyen las economías ilícitas al deterioro ambiental y al fortalecimiento de los grupos armados en América Latina?"},
    {"query_id": "q036", "texto_query": "¿Qué actividades económicas ilegales convergen con la explotación de recursos naturales en zonas de conflicto?"},
    {"query_id": "q037", "texto_query": "¿De qué manera el crimen organizado logra fortalecer su capacidad de influencia sobre las instituciones del Estado?"},
    {"query_id": "q038", "texto_query": "¿Qué innovaciones tácticas recientes han incorporado los grupos armados para aumentar su capacidad operacional?"},
    {"query_id": "q039", "texto_query": "¿Qué capacidades han fortalecido los grupos armados para consolidar y expandir su control territorial?"},
    {"query_id": "q040", "texto_query": "¿Cómo utilizan los grupos armados ilegales la imposición de normas de conducta para ejercer control social sobre la población civil en regiones con baja presencia del Estado?"},
    {"query_id": "q041", "texto_query": "¿Qué corredores de movilidad y territorios estratégicos son priorizados por los Grupos Armados Organizados (GAO), Grupos Armados Organizados Residuales (GAOR) y Grupos Delictivos Organizados (GDO) para asegurar sus economías ilícitas y fortalecer su control territorial?"},
    {"query_id": "q042", "texto_query": "¿De qué manera la minería ilegal de oro financia el fortalecimiento y la expansión territorial de los Grupos Armados Organizados (GAO), Grupos Armados Organizados Residuales (GAOR) y Grupos Delictivos Organizados (GDO) en departamentos como Chocó, Antioquia y Bolívar?"},
    {"query_id": "q043", "texto_query": "¿De qué manera el narcotráfico financia el fortalecimiento y la expansión territorial de los Grupos Armados Organizados (GAO), Grupos Armados Organizados Residuales (GAOR) y Grupos Delictivos Organizados (GDO) en departamentos como Norte de Santander, Arauca, Córdoba y el Cauca?"},
    {"query_id": "q044", "texto_query": "¿Qué factores incrementan el riesgo de reclutamiento, uso y utilización de niños, niñas y adolescentes en zonas de disputa entre grupos armados organizados?"},
    {"query_id": "q045", "texto_query": "¿Cómo afectan las disputas entre grupos criminales la seguridad de la población civil y el incremento de los homicidios selectivos?"},
    {"query_id": "q046", "texto_query": "¿Qué minerales estratégicos están siendo utilizados por los grupos armados organizados en Colombia y la región para financiar el conflicto?"},
    {"query_id": "q047", "texto_query": "¿Cómo la disputa por el control de rutas aéreas incrementa el tráfico de narcóticos, armas y mercancías de contrabando?"},
    {"query_id": "q048", "texto_query": "¿Cómo influyen las dinámicas de exploración y explotación petrolera en la inserción y fortalecimiento de grupos armados que buscan controlar rentas derivadas de recursos estratégicos?"},
    {"query_id": "q049", "texto_query": "¿De qué manera el accionar de grupos armados interfiere y obstaculiza el desarrollo de los procesos de restitución de tierras?"},
    {"query_id": "q050", "texto_query": "¿Cómo adaptan las organizaciones criminales sus estrategias de control territorial frente a las transformaciones políticas, ambientales, tecnológicas y económicas en América Latina?"}
]

# =====================================================================
# 2. PROCESAMIENTO Y CHUNKING (Sección 3)
# =====================================================================
def contar_palabras(texto):
    return len(texto.split())

def fragmentar_documento(doc, max_palabras=250, overlap_oraciones=1):
    """
    Aplica Chunking semántico respetando el límite estricto de 250 palabras 
    y el requisito obligatorio de completitud lingüística (cortes solo en oraciones).
    """
    texto_limpio = doc.get("texto_limpio", "")
    if not texto_limpio:
        return []
        
    oraciones = nltk.sent_tokenize(texto_limpio)
    fragmentos = []
    chunk_actual = []
    palabras_actuales = 0
    
    for oracion in oraciones:
        p_oracion = contar_palabras(oracion)
        if palabras_actuales + p_oracion > max_palabras and chunk_actual:
            fragmentos.append(" ".join(chunk_actual))
            chunk_actual = chunk_actual[-overlap_oraciones:] if overlap_oraciones > 0 else []
            palabras_actuales = sum(contar_palabras(o) for o in chunk_actual)
            
        chunk_actual.append(oracion)
        palabras_actuales += p_oracion
        
    if chunk_actual:
        fragmentos.append(" ".join(chunk_actual))
        
    return fragmentos

def preparar_base_conocimiento(archivos_jsonl):
    print("\n📦 Procesando y aplicando Chunking a la Base de Conocimiento...")
    almacen_metadata = []
    
    for archivo in archivos_jsonl:
        if os.path.exists(archivo):
            with open(archivo, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    doc = json.loads(line)
                    
                    doc_id = doc.get("doc_id", "DOC-UNKNOWN")
                    fuente = doc.get("fuente", archivo)
                    formato = doc.get("formato", "desconocido")
                    fenomeno = doc.get("fenomeno", 1)
                    
                    # Generar chunks respetando reglas
                    chunks_texto = fragmentar_documento(doc, max_palabras=250, overlap_oraciones=1)
                    
                    for pos, texto_chunk in enumerate(chunks_texto):
                        chunk_id = f"{doc_id}-chunk-{pos:03d}"
                        num_tokens = len(texto_chunk.split()) # Aproximación estándar de tokens
                        
                        # Metadata obligatoria (Tabla 1)
                        meta_chunk = {
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "fuente": fuente,
                            "formato": formato,
                            "fenomeno": fenomeno,
                            "posicion": pos,
                            "num_tokens": num_tokens,
                            "texto": texto_chunk
                        }
                        almacen_metadata.append(meta_chunk)
            print(f"   [OK] {archivo} fragmentado con éxito.")
        else:
            print(f"   [WARNING] Archivo {archivo} no encontrado.")
            
    print(f"✅ Total de fragmentos generados para indexación: {len(almacen_metadata)}")
    return almacen_metadata

# =====================================================================
# 3. CARGA DE GROUND TRUTH
# =====================================================================
def normalizar_nombre(nombre):
    """Normaliza un nombre de archivo para poder compararlo entre esquemas
    distintos (el DOCUMENTO del ground truth vs. el Nombre estandarizado
    del índice maestro): minúsculas, sin extensión, sin separadores."""
    nombre = str(nombre).strip().lower()
    nombre = re.sub(r'\.(pdf|json|csv|xlsx)$', '', nombre)
    nombre = re.sub(r'[^a-z0-9]+', '-', nombre)
    return nombre.strip('-')


def construir_crosswalk_doc_id(ruta_indice_maestro):
    """Construye un mapa nombre_de_archivo -> doc_id canónico a partir de la
    hoja 'Inventario de Archivos' del índice maestro (Indice_Datos_Codefest.xlsx).
    Esto es necesario porque el DOCUMENTO del ground truth NO usa el mismo
    esquema de doc_id que genera este pipeline (ver diagnóstico: 'DOC-0296'
    vs. el doc_id real 'F3-MAPPOEA-020')."""
    import openpyxl
    wb = openpyxl.load_workbook(ruta_indice_maestro, data_only=True, read_only=True)
    ws = wb['Inventario de Archivos']
    crosswalk_sin_prefijo = {}
    crosswalk_completo = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        fenomeno, observatorio, codigo, doc_id, nombre_std, carpeta, tipo = row
        if not nombre_std or not doc_id:
            continue
        crosswalk_completo[normalizar_nombre(nombre_std)] = doc_id
        if codigo:
            sin_prefijo = re.sub(rf'^{re.escape(codigo)}_', '', nombre_std, flags=re.IGNORECASE)
            crosswalk_sin_prefijo[normalizar_nombre(sin_prefijo)] = doc_id
    return crosswalk_sin_prefijo, crosswalk_completo


def resolver_doc_id(celda_documento, crosswalk_sin_prefijo, crosswalk_completo):
    """La celda DOCUMENTO trae el nombre de archivo y el id ad-hoc del anotador
    separados por un salto de línea (p.ej. 'Informe-Semestral-36-MAPPOEA-1.pdf\\nDOC-0296-chunk-0052').
    Solo el nombre de archivo es confiable para cruzar contra el índice maestro."""
    if pd.isna(celda_documento):
        return None
    nombre_archivo = str(celda_documento).split('\n')[0].strip()
    clave = normalizar_nombre(nombre_archivo)
    return crosswalk_sin_prefijo.get(clave) or crosswalk_completo.get(clave)


def detectar_fila_header(ruta_excel, hoja, max_filas_busqueda=10):
    """Algunas hojas (p.ej. F1) tienen una fila de título antes del encabezado
    real, lo que hace que pandas lea mal las columnas (Unnamed: 0, Unnamed: 1...).
    Esto busca la fila donde realmente aparece 'PREGUNTA' y la usa como header."""
    df_raw = pd.read_excel(ruta_excel, sheet_name=hoja, header=None, nrows=max_filas_busqueda)
    for i, row in df_raw.iterrows():
        valores = [str(v).strip().upper() for v in row if pd.notna(v)]
        if 'PREGUNTA' in valores:
            return i
    return 0


def cargar_ground_truth_excel(ruta_excel, ruta_indice_maestro):
    print(f"📖 Extrayendo Ground Truth desde: {ruta_excel}...")
    if not os.path.exists(ruta_excel):
        print(f"⚠️ Aviso: {ruta_excel} no encontrado.")
        return {}
    if not os.path.exists(ruta_indice_maestro):
        print(f"⚠️ Aviso: índice maestro {ruta_indice_maestro} no encontrado; no se podrán resolver doc_id.")
        return {}

    crosswalk_sin_prefijo, crosswalk_completo = construir_crosswalk_doc_id(ruta_indice_maestro)

    xls = pd.ExcelFile(ruta_excel)
    ground_truth_dict = {}
    total_docs_sin_resolver = 0

    for hoja in xls.sheet_names:
        fila_header = detectar_fila_header(ruta_excel, hoja)
        df = pd.read_excel(xls, sheet_name=hoja, header=fila_header)

        if 'PREGUNTA' not in df.columns:
            print(f"   [AVISO] Hoja '{hoja}' no tiene columna PREGUNTA reconocible, se omite.")
            continue

        # Forward-fill: en el Excel original PREGUNTA es una celda combinada
        # que cubre varias filas de evidencia (FRAGMENTO/DOCUMENTO distintos
        # para la misma pregunta). Sin este paso, esas filas de continuación
        # se pierden por completo al hacer dropna(subset=['PREGUNTA']).
        df['PREGUNTA'] = df['PREGUNTA'].ffill()
        df_validas = df.dropna(subset=['PREGUNTA'])

        agrupado = df_validas.groupby('PREGUNTA').agg({
            'FRAGMENTO': lambda x: list(x.dropna().astype(str).unique()),
            'DOCUMENTO': lambda x: list(x.dropna().astype(str).unique())
        }).reset_index()

        for _, row in agrupado.iterrows():
            pregunta_limpia = row['PREGUNTA'].strip()
            doc_ids_resueltos = []
            for celda in row['DOCUMENTO']:
                resuelto = resolver_doc_id(celda, crosswalk_sin_prefijo, crosswalk_completo)
                if resuelto:
                    doc_ids_resueltos.append(resuelto)
                else:
                    total_docs_sin_resolver += 1
                    print(f"   [AVISO] No se pudo resolver doc_id para: {celda.splitlines()[0]!r}")

            ground_truth_dict[pregunta_limpia] = {
                "doc_ids_relevantes": doc_ids_resueltos,
                # Texto libre anotado a mano (columna FRAGMENTO), no chunk_id;
                # calcular_ndcg_10 compara por solapamiento de texto, no por id.
                "chunk_ids_relevantes": row['FRAGMENTO']
            }
        print(f"   [OK] Hoja '{hoja}': header detectado en fila {fila_header}, {len(agrupado)} preguntas cargadas.")

    print(f"✅ Ground Truth cargado: {len(ground_truth_dict)} preguntas totales "
          f"({total_docs_sin_resolver} referencias DOCUMENTO sin resolver).")
    return ground_truth_dict

# =====================================================================
# 4. FÓRMULAS DE EVALUACIÓN OFICIAL (Sección 10.2)
# =====================================================================
def _texto_normalizado(texto):
    return re.sub(r'\s+', ' ', str(texto).lower()).strip()

def _fragmento_coincide_con_chunk(fragmento_texto, chunk_texto, umbral=0.5):
    """El ground truth marca relevancia con un excerto de texto libre (columna
    FRAGMENTO) anotado a mano, no con nuestro chunk_id, así que la única forma
    de saber si un chunk generado por el pipeline corresponde a ese excerto es
    comparar el contenido: por inclusión directa, o por similitud de texto
    cuando el corte de oraciones no coincide exactamente con el fragmento anotado."""
    frag = _texto_normalizado(fragmento_texto)
    chunk = _texto_normalizado(chunk_texto)
    if not frag or not chunk:
        return False
    if frag in chunk or chunk in frag:
        return True
    return SequenceMatcher(None, frag, chunk).ratio() >= umbral

def calcular_ndcg_10(ranking_chunk_textos, fragmentos_relevantes):
    fragmentos_relevantes = [f for f in fragmentos_relevantes if _texto_normalizado(f)]
    if not fragmentos_relevantes: return 0.0
    top_10 = ranking_chunk_textos[:10]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, chunk_texto in enumerate(top_10)
        if any(_fragmento_coincide_con_chunk(frag, chunk_texto) for frag in fragmentos_relevantes)
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(fragmentos_relevantes), 10)))
    return dcg / idcg if idcg > 0 else 0.0

def calcular_f1_3(ranking_doc_ids, doc_ids_relevantes):
    relevantes = set(doc_ids_relevantes)
    if not relevantes: return 0.0
    top_3 = []
    for d in ranking_doc_ids:
        if d not in top_3: top_3.append(d)
        if len(top_3) == 3: break
    interseccion = len(set(top_3).intersection(relevantes))
    if interseccion == 0: return 0.0
    p_3 = interseccion / 3.0
    r_3 = interseccion / min(len(relevantes), 3.0)
    return (2 * p_3 * r_3) / (p_3 + r_3)

# =====================================================================
# 5. PIPELINE DE EVALUACIÓN DE LOS 6 MODELOS SOLICITADOS
# =====================================================================
def ejecutar_benchmark_6_modelos(archivos_jsonl, ruta_excel_gt, ruta_indice_maestro):
    # Los 6 modelos exactos solicitados
    modelos = [
        {"nombre": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "prefijo_q": "", "prefijo_p": ""},
        {"nombre": "intfloat/multilingual-e5-small", "prefijo_q": "query: ", "prefijo_p": "passage: "},
        {"nombre": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "prefijo_q": "", "prefijo_p": ""},
        {"nombre": "BAAI/bge-m3", "prefijo_q": "", "prefijo_p": ""},  # bge-multilingual-base-v1.5 no existe en HF; bge-m3 es el modelo multilingüe real de BAAI y no usa prefijo de instrucción
        {"nombre": "intfloat/multilingual-e5-base", "prefijo_q": "query: ", "prefijo_p": "passage: "}
    ]
    
    ground_truth_dict = cargar_ground_truth_excel(ruta_excel_gt, ruta_indice_maestro)
    almacen_metadata = preparar_base_conocimiento(archivos_jsonl)
    
    if not almacen_metadata:
        print("❌ Error crítico: No hay fragmentos procesados.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Aceleración por hardware activa: {device.upper()}")

    resultados_modelos = []

    for config in modelos:
        nombre = config["nombre"]
        print(f"\n=======================================================")
        print(f"⚙️ Procesando Modelo: {nombre}")
        
        try:
            modelo = SentenceTransformer(nombre, device=device)
            dim = modelo.get_embedding_dimension()
            
            textos_p = [f"{config['prefijo_p']}{c['texto']}" for c in almacen_metadata]
            t0 = time.time()
            vectores = modelo.encode(textos_p, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
            print(f"⚡ Vectorización completada en {time.time() - t0:.2f} s")
            
            indice_faiss = faiss.IndexFlatIP(dim)
            indice_faiss.add(np.array(vectores, dtype=np.float32))
            
            entregable_jsonl = []
            ndcg_lista, f1_lista = [], []
            
            for consulta in CONSULTAS_OFICIALES:
                texto_query_limpio = consulta['texto_query'].strip()
                texto_q_encode = f"{config['prefijo_q']}{texto_query_limpio}"
                vector_q = modelo.encode([texto_q_encode], normalize_embeddings=True)
                
                scores, indices = indice_faiss.search(np.array(vector_q, dtype=np.float32), 100)
                
                fragmentos_recuperados = []
                ranking_chunks = []
                docs_agrupados = {} 
                
                for i, idx in enumerate(indices[0]):
                    chunk_meta = almacen_metadata[idx]
                    score = float(scores[0][i])
                    
                    ranking_chunks.append(chunk_meta["texto"])
                    
                    if len(fragmentos_recuperados) < 10:
                        fragmentos_recuperados.append({
                            "rank": len(fragmentos_recuperados) + 1,
                            "chunk_id": chunk_meta["chunk_id"],
                            "doc_id": chunk_meta["doc_id"],
                            "text": chunk_meta["texto"]
                        })
                    
                    doc_id_actual = chunk_meta["doc_id"]
                    if doc_id_actual not in docs_agrupados:
                        docs_agrupados[doc_id_actual] = score
                        
                top_3_docs = sorted(docs_agrupados.items(), key=lambda x: x[1], reverse=True)[:3]
                documentos_formateados = [
                    {"rank": r + 1, "doc_id": doc_id} 
                    for r, (doc_id, _) in enumerate(top_3_docs)
                ]
                
                entregable_jsonl.append({
                    "query_id": consulta["query_id"],
                    "documents": documentos_formateados,
                    "fragments": fragmentos_recuperados
                })
                
                if texto_query_limpio in ground_truth_dict:
                    gt = ground_truth_dict[texto_query_limpio]
                    ranking_docs_final = [doc for doc, _ in sorted(docs_agrupados.items(), key=lambda x: x[1], reverse=True)]
                    
                    ndcg = calcular_ndcg_10(ranking_chunks, gt["chunk_ids_relevantes"])
                    f1 = calcular_f1_3(ranking_docs_final, gt["doc_ids_relevantes"])
                    
                    ndcg_lista.append(ndcg)
                    f1_lista.append(f1)
                    
            nombre_archivo = f"resultados_{nombre.split('/')[-1]}.jsonl"
            with open(nombre_archivo, 'w', encoding='utf-8') as fout:
                for res in entregable_jsonl:
                    fout.write(json.dumps(res, ensure_ascii=False) + '\n')
                    
            promedio_ndcg = np.mean(ndcg_lista) if ndcg_lista else 0.0
            promedio_f1 = np.mean(f1_lista) if f1_lista else 0.0
            
            resultados_modelos.append({
                "modelo": nombre,
                "ndcg_10": promedio_ndcg,
                "f1_3": promedio_f1
            })
            print(f"✅ Generado: {nombre_archivo} | NDCG@10: {promedio_ndcg:.4f} | F1@3: {promedio_f1:.4f}")
            
        except Exception as e:
            print(f"❌ Error procesando el modelo {nombre}: {e}")

    # =====================================================================
    # 6. LEADERBOARD UNIFICADO (CONTEO DE BORDA - Sección 11.2)
    # =====================================================================
    if not resultados_modelos:
        print("⚠️ No hay resultados exitosos para calcular el Conteo de Borda.")
        return

    print("\n🏆 CALCULANDO LEADERBOARD UNIFICADO (CONTEO DE BORDA)...")
    N = len(resultados_modelos)
    
    resultados_modelos.sort(key=lambda x: x["ndcg_10"], reverse=True)
    for pos, res in enumerate(resultados_modelos): res["puntos_ndcg"] = N - (pos + 1)
        
    resultados_modelos.sort(key=lambda x: x["f1_3"], reverse=True)
    for pos, res in enumerate(resultados_modelos):
        res["puntos_f1"] = N - (pos + 1)
        res["borda_total"] = res["puntos_ndcg"] + res["puntos_f1"]
        
    resultados_modelos.sort(key=lambda x: (x["borda_total"], x["ndcg_10"]), reverse=True)
    
    print(f"\n{'MODELO':<55} | {'NDCG@10':<8} | {'F1@3':<8} | {'PUNTOS BORDA'}")
    print("-" * 95)
    for res in resultados_modelos:
        indicador = "👑 GANADOR" if res == resultados_modelos[0] else "  "
        print(f"{indicador:<10} {res['modelo']:<44} | {res['ndcg_10']:.4f}   | {res['f1_3']:.4f} | {res['borda_total']}")

if __name__ == "__main__":
    # Ajusta las rutas relativas o absolutas a tus carpetas reales si es necesario
    archivos_jsonl_limpios = ["../data/clean/f1_documentos.jsonl", "../data/clean/f2_documentos.jsonl", "../data/clean/f3_documentos.jsonl"]
    archivo_excel_gt = "../data/raw/F3_Dinamicas_Territoriales/FASE ORDENADA CODEFEST.xlsx"
    archivo_indice_maestro = "../data/raw/Indice_Datos_Codefest.xlsx"  # ajusta si tu copia local vive en otra ruta

    ejecutar_benchmark_6_modelos(archivos_jsonl_limpios, archivo_excel_gt, archivo_indice_maestro)
