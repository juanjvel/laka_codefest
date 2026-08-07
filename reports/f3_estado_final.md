# Estado final — Extracción F3 (Dinámicas Territoriales)

Verificado el 2026-08-06 contra el estado real en disco de `data/clean/` (no contra logs
ni corridas anteriores). CODEFEST AD ASTRA 2026 — Etapa 1.

## 1. Conteos finales

| Métrica | Valor |
|---|---|
| Filas en inventario F3 (`Indice_Datos_Codefest.xlsx`) | 888 |
| doc_id únicos esperados (tras dedup, p.ej. duplicados de Amazon Underworld) | 828 |
| Documentos limpios (`f3_documentos.jsonl`) | 819 |
| Pendientes (`f3_pendientes.jsonl`) | 9 |
| **Suma procesados + pendientes** | **828 = 828 esperados ✅** |
| doc_id duplicados en `f3_documentos.jsonl` | 0 ✅ |
| Líneas JSON inválidas en `f3_documentos.jsonl` | 0 ✅ |

Todo doc_id esperado del inventario está contabilizado: no hay documentos "perdidos" entre
corridas (verificado por diferencia de conjuntos, no por conteo aproximado).

## 2. Corrección de acentos en vector tiles (.pbf)

**Problema original:** la librería `mapbox_vector_tile` devolvía los `string_value` de los
`.pbf` con mojibake (p.ej. `Urucar\xc3\xa1` se decodificaba mal) pese a que los bytes crudos
del archivo ya estaban en UTF-8 válido.

**Solución aplicada:** se reemplazó por `blackboxprotobuf` (decodificación de bajo nivel,
sin esquema fijo), replicando a mano el esquema Tile/Layer/Feature/Value de la especificación
Mapbox Vector Tile (`src/extraccion.py::extraer_otro`).

**Verificación en los 13 `.pbf` reales de `f3_documentos.jsonl`:**
- 0 ocurrencias de `U+FFFD` (carácter de reemplazo, señal de mojibake) en `texto_limpio`.
- Muestra visual confirmando acentos y ñ/ç correctos: *Urucará*, *Pará*, *Maranhão*,
  *Huanca Sancos*, *Ayacucho*, *Quito*, *Pichincha* — todos los 13 provienen de
  `Amazon_Underworld`.

## 3. Mitigación de OneDrive / Windows

`os.replace()` puede fallar en Windows con `PermissionError` (WinError 5) si el archivo
destino tiene un handle abierto momentáneamente por un antivirus, el indexador de Windows,
o un cliente de sincronización como OneDrive (relevante si el repo llegara a vivir dentro de
una carpeta sincronizada). Se implementó `_reemplazar_con_reintento()`
(`src/extraccion.py`): hasta 6 reintentos con backoff (1–2s por intento, ~9s acumulados),
usado en la segunda pasada que marca duplicados de `.pbf` por hash.

Adicionalmente, `LockFenomeno` usa creación exclusiva de archivo (`O_CREAT|O_EXCL`) para
impedir que dos corridas escriban al mismo JSONL/checkpoint a la vez — esto ya había
corrompido datos una vez en producción (duplicados + `PermissionError` concurrente) antes de
añadir el lock.

## 4. Pendientes finales — aceptados y definitivos (9)

No requieren más intentos de reprocesamiento; se documentan como exclusiones/errores
definitivos, no como trabajo pendiente:

| Motivo | Cantidad | doc_id | Explicación |
|---|---|---|---|
| `catalogo_excluido` | 6 | `CEOBS_catalog-2.json`, `MAPPOEA_mapp-catalog.csv`, `MAPPOEA_mapp-catalog.json`, `RESDAL_catalog-2.csv`, `RESDAL_catalog-2.json`, `SIPRI_catalog-2.json` | Exclusión intencional: son catálogos/registros de metadata, no documentos de contenido (`PATRON_CATALOGO`). |
| `error_extraccion` | 2 | `SIPRI_22136.pdf`, `SIPRI_hsrc20lmip20report20320growth20employment20and20skills-1.pdf` | PDFs corruptos: `PDFSyntaxError: No /Root object!` — el archivo descargado no es un PDF válido. No es un error de memoria ni de OCR; requeriría volver a descargar el archivo original desde SIPRI, fuera del alcance de esta etapa. |
| `bulk_dump_json` | 1 | `AMAZONUW_tiles-index.json` | Intencional: la raíz del JSON es una lista con 262 registros (índice de tiles), no un documento individual — se documenta en vez de fusionar 262 registros en un solo texto. |

## 5. Campo `idioma_detectado` — nota de calidad de datos

363/819 documentos (44%) no tienen `idioma_detectado` (el campo se omite del JSON cuando es
`None`, no aparece como `null`). Auditado con evidencia completa (no solo muestra): **el
100% de los 363 corresponde a un único patrón** — JSON de la fuente `Alertas_Tempranas` cuyo
único campo de texto extraído es la palabra `"Mapa"` (4 caracteres), muy por debajo del
umbral de 20 caracteres de `detectar_idioma()`. Longitud máxima entre los 363: 4 caracteres.
Cero casos de texto largo/coherente mal detectado. No requiere ajuste del umbral de
`langdetect` ni reprocesamiento.

## 6. Optimizaciones de memoria aplicadas al pipeline

Necesarias para completar la corrida de forma estable tras cuelgues por agotamiento de RAM
(no por un bug lógico) en documentos pesados de CEOBS/RESDAL/MAPP_OEA:

- **Streaming real**: cada documento se escribe (append + flush) al JSONL correspondiente
  inmediatamente después de procesarse, nunca se acumula el corpus completo en memoria.
- **OCR de PDF página por página**: renderizado vía `pdf2image` con `first_page`/`last_page`
  (nunca `convert_from_path()` del documento completo), `dpi=100` (bajado de 150 tras los
  `MemoryError` reproducidos), y `img.close()` + `del` + `gc.collect()` explícito tras cada
  página antes de pasar a la siguiente.
- **Límite de memoria del proceso** (`verificar_memoria_o_pausar()` en
  `src/extraccion.py`, vía `psutil` — portable a Windows, donde `resource` de Unix no
  aplica): antes de cada documento y de cada página OCR se revisa
  `psutil.virtual_memory().percent`; por encima de 85% pausa 3s y fuerza `gc.collect()`; si
  sigue por encima de 92% tras la pausa, el documento se salta a pendientes con motivo
  `memoria_insuficiente` (reintentable automáticamente en la siguiente corrida) en vez de
  arriesgar un `MemoryError` que tumbe el proceso.
- **Procesamiento estrictamente secuencial**: sin `ThreadPoolExecutor` ni
  `multiprocessing` en ningún punto del pipeline — el paralelismo es lo que dispara picos de
  memoria simultáneos.
- **Batching pequeño** (`src/reprocesar_pendientes.py`): lotes de 10 documentos, con pausa
  de 2s y `gc.collect()` entre lotes, para reprocesamientos dirigidos de pendientes.

## 7. Reproducción desde cero

**Dependencias** (`requirements-extraccion.txt`):
```
openpyxl==3.1.5
pdfplumber==0.11.4
pytesseract==0.3.13
pdf2image==1.17.0
Pillow==11.0.0
chardet==5.2.0
blackboxprotobuf==1.0.1
langdetect==1.0.9
psutil==6.1.0
```
Requiere además los binarios de sistema **Tesseract OCR** y **Poppler** en el `PATH` (no se
instalan vía pip).

**Comando para la corrida completa de F3:**
```bash
pip install -r requirements-extraccion.txt
python src/extraccion.py --fenomenos F3
```

**Comando para reprocesar pendientes dirigidos** (p.ej. solo los de un motivo específico):
```bash
python src/reprocesar_pendientes.py --motivo error_extraccion
```

**Tiempo estimado:** ~2–4 horas para los 828 documentos de F3 en una máquina estándar
(dominado por OCR de PDFs escaneados vía Tesseract; el resto de tipos —JSON/CSV/Excel/
Texto/.pbf— es prácticamente instantáneo). El proceso es reanudable: si se interrumpe,
relanzar el mismo comando continúa desde el checkpoint (`f3_procesados.txt`) sin reprocesar
lo ya completado.

**Validación post-corrida:** `python src/extraccion.py --fenomenos F3` termina imprimiendo
automáticamente la comparación contra la hoja "Resumen por Fenomeno" del Excel
(`validar_conteos`), y este reporte puede regenerarse manualmente repitiendo los 5 chequeos
de la sección de verificación de esta corrida.
