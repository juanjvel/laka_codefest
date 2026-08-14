# CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento

**Equipo:** LAKA

Entregable de la Etapa 1 del reto CODEFEST AD ASTRA 2026: una base de conocimiento vectorial
(FAISS) construida a partir del corpus documental provisto por ADL, y un módulo de recuperación
puramente vectorial (sin modelos generativos) que responde las 50 consultas de evaluación
(q001–q050).

## Estructura de `entrega/`

```
entrega/
  generador.py                                 # único script ejecutable del entregable
  resultados.jsonl                              # salida ya generada (50 líneas, q001-q050)
  informe_tecnico.pdf                           # documento técnico (decisiones de diseño)
  base_vectorial/
    encoder_multilingual-e5-small/
      index.faiss                               # índice FAISS (faiss.write_index)
      metadata.jsonl                             # metadata por chunk, Tabla 1 del spec
    grafo/
      grafo.graphml                              # Grafo del proyecto como bonus
```

Para el detalle de las decisiones de diseño (estrategia de chunking y su justificación,
encoder(s) seleccionado(s) y criterios de elección, tipo de índice FAISS empleado) ver
`informe_tecnico.pdf`.

> **Nota (Git LFS):** `index.faiss` (~485 MB) y `metadata.jsonl` (~320 MB) se versionan con Git
> LFS. Antes de tocar el entorno de Python, sigue la sección 1 para asegurarte de tener el
> contenido real de esos archivos y no solo sus punteros.

## Requisitos

- **Python ≥ 3.10** (el entorno de referencia usado para generar `resultados.jsonl` corre en
  Python 3.10.14; cualquier 3.10+ funciona).
- ~2 GB libres en disco para las dependencias (PyTorch + Transformers son los paquetes más
  pesados) y ~1 GB adicional de RAM/disco para cargar `index.faiss` + `metadata.jsonl`.
- Conexión a internet la primera vez que se ejecuta el script: `sentence-transformers` descarga
  el encoder `intfloat/multilingual-e5-small` desde HuggingFace, y `nltk` descarga los modelos
  `punkt`/`punkt_tab` la primera vez que se usan.

## 1. Instalar y configurar Git LFS

`entrega/base_vectorial/encoder_multilingual-e5-small/` contiene dos archivos grandes
(`index.faiss` ~485 MB y `metadata.jsonl` ~320 MB) versionados con [Git LFS](https://git-lfs.com)
en vez de con Git normal. Git LFS reemplaza esos archivos en el historial por un puntero de
texto de unos pocos bytes y descarga el contenido real por separado, solo cuando hace falta.
Esto es necesario porque Git no está diseñado para versionar binarios de cientos de MB
eficientemente.

### 1.1 Instalar el cliente de Git LFS

Se instala una sola vez por máquina, no por repositorio.

```bash
# Debian / Ubuntu
sudo apt install git-lfs

# macOS (Homebrew)
brew install git-lfs

# Windows (winget)
winget install GitHub.GitLFS
```

Si tu gestor de paquetes no lo tiene, descarga el instalador desde
[git-lfs.com](https://git-lfs.com). Verifica la instalación con:

```bash
git lfs version
```

### 1.2 Activar el hook de Git LFS en tu configuración de usuario

También es un paso único por máquina (registra los hooks de Git que interceptan los archivos
LFS en cualquier repositorio que clones o uses después):

```bash
git lfs install
```

### 1.3 Obtener el contenido real de `index.faiss` y `metadata.jsonl`

Si clonas el repositorio **después** de haber corrido `git lfs install`, `git clone` ya descarga
el contenido real de los archivos LFS automáticamente — no necesitas nada adicional.

Si el repositorio ya estaba clonado antes de instalar Git LFS (o si `index.faiss` /
`metadata.jsonl` pesan solo un par de cientos de bytes, señal de que solo tienes el puntero),
descarga el contenido real con:

```bash
git lfs pull
```

### 1.4 Verificar que los archivos son el contenido real y no punteros

```bash
git lfs ls-files                # lista qué archivos del repo están gestionados por LFS
ls -lh entrega/base_vectorial/encoder_multilingual-e5-small/
```

`index.faiss` debe pesar ~485 MB y `metadata.jsonl` ~320 MB. Si en vez de eso ves un archivo de
~130 bytes con contenido como `version https://git-lfs.github.com/spec/v1 ...`, Git LFS no está
instalado/activado o `git lfs pull` no se corrió — repite los pasos 1.1 a 1.3.

`generador.py` (paso 3) carga `index.faiss` con `faiss.read_index()`, que espera el binario real;
si solo tienes el puntero de texto, fallará al cargar el índice.

## 2. Crear el entorno virtual

Usa el gestor que prefieras; ambas rutas quedan documentadas.

### Opción A — `venv` (estándar de la librería)

```bash
cd entrega   # o la carpeta donde hayas colocado este entregable

python3 --version            # confirma que sea >= 3.10 antes de continuar
python3 -m venv .venv
source .venv/bin/activate    # en Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
```

### Opción B — `conda`

```bash
conda create -n codefest-etapa1 python=3.10 -y
conda activate codefest-etapa1
python -m pip install --upgrade pip
```

## 3. Instalar dependencias

`generador.py` solo necesita `faiss`, `nltk` y `sentence-transformers` (que a su vez instala
`torch` y `transformers`). `pandas`/`openpyxl` solo son necesarios si el archivo de consultas
(`--consultas`) viene en `.csv` o `.xlsx` en vez de `.jsonl`/`.json`.

Si el evaluador necesita cargar el grafo usando Python (por ejemplo, para evaluar la topología o
integrarlo con el FAISS), es estándar usar la librería `networkx`. Agrega esto a tus
instrucciones de `pip`:

```bash
pip install faiss-cpu nltk sentence-transformers networkx
pip install pandas openpyxl   # opcional: solo si usarás consultas en .csv/.xlsx
```

Si dispones de GPU NVIDIA y quieres acelerar la inferencia del encoder, instala la variante GPU
de FAISS en vez de `faiss-cpu` (la build de PyTorch con soporte CUDA la resuelve
automáticamente `pip install torch` según tu sistema):

```bash
pip install faiss-gpu-cu12 nltk sentence-transformers
```

Descarga los recursos de NLTK que usa `generador.py` (tokenización de oraciones para respetar el
límite de 250 palabras por fragmento, Sección 9.2.1 del spec):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Versiones de referencia (entorno donde se generó `resultados.jsonl`)

No son un requisito estricto — cualquier versión reciente y mutuamente compatible de estos
paquetes funciona — pero se documentan para reproducibilidad exacta:

| Paquete | Versión |
|---|---|
| Python | 3.10.14 |
| faiss | 1.14.1 |
| nltk | 3.10.0 |
| sentence-transformers | 5.7.0 |
| torch | 2.13.0 |
| transformers | 5.15.0 |
| numpy | 2.2.5 |
| pandas | 2.3.3 |
| openpyxl | 3.1.5 |

## 4. Uso de `generador.py`

`generador.py` es el módulo de Recuperación (Sección 8 del spec): toma las 50 consultas de
evaluación, busca en el índice FAISS ya construido, agrega los fragmentos a nivel de documento
(max pooling por defecto, Sección 8.6), aplica el límite de 250 palabras por fragmento (Sección
9.2.1) y escribe `resultados.jsonl` en el formato exacto de la Sección 9.3. **No usa en ningún
punto un modelo generativo** (Sección 8.3): toda la lógica opera sobre el vector de consulta, las
puntuaciones de FAISS y los campos de metadata.

### Comando básico

```bash
cd entrega
python generador.py \
    --indice base_vectorial/encoder_multilingual-e5-small \
    --consultas consultas.jsonl \
    --salida resultados.jsonl
```

Al terminar, el script valida automáticamente el `resultados.jsonl` generado (50 líneas, 3
documentos y 10 fragmentos por consulta, ningún fragmento por encima de 250 palabras) e imprime
el resultado de esa validación en consola.

### Argumentos

| Argumento | Obligatorio | Por defecto | Descripción |
|---|---|---|---|
| `--indice` | sí | — | Carpeta `encoder_<nombre>/` con `index.faiss` y `metadata.jsonl` |
| `--consultas` | sí | — | Archivo con las consultas a resolver (ver formatos abajo) |
| `--salida` | no | `resultados.jsonl` | Ruta de salida en formato JSON Lines |
| `--modelo` | no | `intfloat/multilingual-e5-small` | Encoder de HuggingFace (debe ser el mismo con el que se construyó el índice) |
| `--device` | no | autodetección | `cpu`, `cuda`, `cuda:0`, ... |
| `--k-inicial` | no | `30` | Tamaño de búsqueda inicial en FAISS por consulta |
| `--k-maximo` | no | `200` | Tope al que se duplica `k` si no se alcanzan los 3 documentos / 10 fragmentos requeridos |
| `--agregacion` | no | `max` | Estrategia de agregación fragmento→documento: `max`, `sum` o `mean` (Sección 8.6) |
| `--max-palabras` | no | `250` | Límite de palabras por fragmento (Sección 9.2.1) |

### Formatos aceptados para `--consultas`

Autodetectado por extensión:

- **`.jsonl`**: una línea por consulta, `{"query_id": "q001", "query": "..."}`
- **`.json`**: lista de objetos con las mismas claves, o `{"q001": "texto", ...}`
- **`.csv` / `.xlsx`**: columnas de id y texto (acepta `query_id`/`id`/`consulta_id`/`qid` y
  `query`/`consulta`/`pregunta`/`texto`/`text` como nombres de columna; requiere `pandas` y,
  para `.xlsx`, `openpyxl`)

Si el archivo real de consultas usa nombres de columna distintos, se pueden agregar a las listas
`ALIAS_ID` / `ALIAS_TEXTO` al inicio de `generador.py` — es el único punto de ajuste necesario.

### Ejemplo con parámetros no-default

```bash
python generador.py \
    --indice base_vectorial/encoder_multilingual-e5-small \
    --consultas consultas.jsonl \
    --salida resultados.jsonl \
    --agregacion sum \
    --k-inicial 50 \
    --device cuda
```

## 5. Validar un `resultados.jsonl` ya generado

`generador.py` valida su propia salida al final de la ejecución. Si necesitas revalidar un
archivo por separado (por ejemplo tras editarlo manualmente), puedes reutilizar la misma función
desde un intérprete, estando parado en `entrega/`:

```bash
python -c "from generador import validar_resultados; validar_resultados('resultados.jsonl')"
```

## Restricciones del spec que aplican a este entregable

- **Sin modelos generativos** (decoders tipo GPT/LLaMA/Gemini/Claude) en ninguna etapa de
  recuperación (Sección 8.3): ni reranking, ni resumen, ni expansión de consulta.
- **Completitud lingüística obligatoria** (Sección 3.3): ningún fragmento puede cortar una
  oración a la mitad; los cortes solo ocurren en límites oracionales completos.
- **Mismo encoder para indexar y consultar** (Sección 8.1): el vector de consulta y los vectores
  del índice deben vivir en el mismo espacio semántico — no cambies `--modelo` a menos que
  reconstruyas también `base_vectorial/` con ese mismo encoder.
