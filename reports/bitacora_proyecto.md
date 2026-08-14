# Bitácora del proyecto — CODEFEST AD ASTRA 2026, Etapa 1

Registro acumulado del estado del pipeline, las decisiones tomadas y los
hallazgos que costaron trabajo descubrir. El objetivo es que nadie del equipo
tenga que volver a derivarlos.

Última actualización: 2026-08-13.

---

## 1. Estado del pipeline

| Etapa | Estado | Artefacto |
|---|---|---|
| Extracción y limpieza (F1, F2, F3) | Completa | `data/clean/f{1,2,3}_documentos.jsonl` |
| Chunking semántico | Completo | `data/clean/metadata.jsonl` — **330,416 chunks**, 1,780 doc_id |
| Codificación + índice FAISS | Completo | `entrega/base_vectorial/encoder_multilingual-e5-small/` |
| Módulo de recuperación | Completo | `entrega/generador.py` → `entrega/resultados.jsonl` |
| Informe técnico (máx. 8 pág.) | **Pendiente** | `entrega/informe_tecnico.pdf` |
| Grafo de conocimiento (bonus §7) | **Pendiente** | `entrega/base_vectorial/grafo/grafo.graphml` |

Encoder de producción: `intfloat/multilingual-e5-small`, índice `IndexFlatIP`
de 330,416 vectores L2-normalizados, dim 384. Verificado: identidad
vector↔texto = 1.000000, y la línea N del `metadata.jsonl` corresponde al id
interno N de FAISS.

El detalle de diseño del generador está en `entrega/README.md`.

---

## 2. Caracterización del corpus (medida, no estimada)

Números que conviene tener a mano porque condicionan casi cualquier decisión
aguas abajo:

- **Longitud de chunks**: mín. 1, máx. **exactamente 250** palabras, mediana
  140, p90 228. Ningún chunk supera el límite del spec, así que la división
  post-recuperación nunca se dispara en la práctica.
- **El 31.7 % de los chunks tiene menos de 20 palabras** (104,798): títulos
  sueltos, números de página, `"Overview"`. Vienen sobre todo de PDF (77,495)
  y CSV (26,749).
- **Distribución por fenómeno**: F1 = 200,780 chunks, F2 = 51,859, F3 = 77,777.
  F1 tiene ~4x los chunks de F2 aunque las consultas están balanceadas.
- **Distribución por formato**: csv 160,195, pdf 150,053, pbf 14,549,
  json 4,371, excel 1,230, texto 12, imagen 6. Más de la mitad del corpus son
  formatos tabulares.
- **Skew extremo de documentos**: `F1-AIINDEX-056` (un CSV) aporta **69,162
  chunks**, frente a una mediana de 7 por documento.
- **Los chunks de un documento ocupan líneas consecutivas** del
  `metadata.jsonl`, con `posicion` secuencial desde 0. Por eso el vecino del id
  interno `i` es la línea `i±1` comprobando que el `doc_id` coincida: no hace
  falta ningún índice auxiliar para navegar entre fragmentos contiguos.

---

## 3. Hallazgos del módulo de recuperación

### 3.1 El ground truth local no tiene poder discriminativo

`data/raw/FASE ORDENADA CODEFEST.xlsx` tiene hojas para F1 y F3, **ninguna para
F2**, y solo empareja con **7 de las 50 consultas**.

Un barrido de 8 configuraciones (fórmula de agregación, tamaño del bonus,
profundidad de búsqueda, tope de fragmentos por documento) dio **NDCG@10
idéntico en las ocho**: 0.1429, que es exactamente 1/7 — solo q004 acierta y
las otras seis dan cero, hagamos lo que hagamos. El F1@3 solo tomó dos valores
(0.1190 y 0.0714), y la diferencia es una única consulta (q036) entrando o
saliendo del top-3.

**Conclusión operativa**: no tiene sentido seguir probando parámetros. Lo que
desbloquea el trabajo es **anotar más consultas, empezando por las 16 de F2 que
hoy no tienen ninguna**.

### 3.2 El cuello de botella es la agregación, no la recuperación

Buscando a profundidad 20,000, los documentos del ground truth aparecen así:

| Consulta | doc del GT | primer chunk en rank |
|---|---|---|
| q034 | F3-MAPPOEA-014 | **7** |
| q036 | F3-MAPPOEA-021 | **9** |
| q033 | F3-MAPPOEA-020 | **14** |
| q036 | F3-MAPPOEA-013 | 19 |
| q002 | F1-DAIO-030 | 115 |
| q035 | F3-MAPPOEA-015 | 379 |
| q003 | F1-DAIO-023 | 2013 |

En la mayoría de casos el documento correcto está muy bien rankeado **a nivel
de chunk** y aun así no entra al top-3 de documentos. El encoder y el chunking
están haciendo su trabajo; el paso que pierde la señal es la agregación
chunk→documento.

### 3.3 Los scores coseno de E5 tienen rango muy estrecho dentro de un pool

Medido sobre el índice real: dentro del top-300 de una consulta los scores van
de ~0.873 a ~0.839, un **rango total de 0.034**.

Esto invalida cualquier intuición basada en "la escala coseno va de 0.7 a 0.9".
Concretamente, hizo que la fórmula de agregación
`max + min(0.05, 0.01·(n−1))` no se comporte como max pooling con desempate,
sino como:

1. número de chunks de apoyo en el pool (saturado a 6 por el tope del bonus);
2. score máximo, solo para desempatar.

Es decir, un **conteo acotado de evidencia**. Funciona en la práctica (el CSV de
69k chunks no aparece ni una vez en los 150 slots de documentos), pero **el
informe técnico debe describirla así y no como max pooling**. Cualquier término
aditivo que se añada a un score coseno debe dimensionarse contra este rango de
0.034, no contra el rango teórico [-1, 1].

### 3.4 El ground truth se empareja por contenido, no por identificadores

La Sección 10.2.1 del spec dice que la relevancia de un fragmento se juzga por
el **contenido del campo `text`**, y la de un documento por el campo **`fuente`**
(nombre del archivo original), no por el `doc_id` que inventamos nosotros.

Dos consecuencias prácticas:

- Devolver fragmentos con contenido real importa más que la precisión del
  `chunk_id`. De ahí el enriquecimiento de fragmentos cortos.
- Todo `doc_id` que emitamos debe resolver a un `fuente` no vacío en el
  `metadata.jsonl`. Verificado: los 83 `doc_id` emitidos lo hacen.

Además, el **100 % de los excertos anotados en el ground truth son texto limpio
de una sola línea**, mientras que el 90 % de nuestros fragmentos arrastraba
saltos de línea de la extracción de PDF. Por eso el generador normaliza espacios
en el texto de salida: si el evaluador normaliza, da igual; si no normaliza, nos
salva el emparejamiento.

---

## 4. Bugs encontrados en código existente

### `calcular_ndcg_10` de `src/benchmark_modelos.py` puede devolver > 1

Observado: **1.4307** en q004, imposible para un NDCG (rango [0,1]). Acredita
cada posición del top-10 que coincida con *algún* excerto anotado, mientras que
el IDCG asume como máximo `len(fragmentos_gt)` posiciones relevantes. Con un
único excerto anotado y varios fragmentos nuestros parecidos entre sí, el DCG
supera al IDCG.

**Los resultados históricos de NDCG de `benchmark_modelos.py` están inflados.**
`src/evaluar_resultados.py` lo corrige con emparejamiento uno-a-uno (cada
excerto se acredita una sola vez).

### `src/benchmark_modelos.py` no se puede importar sin red

Hace `nltk.download('punkt')` a nivel de módulo. En una máquina sin `nltk_data`
ni acceso a los servidores de NLTK, importarlo **cuelga indefinidamente**. Por
eso `src/evaluar_resultados.py` copia las funciones de métrica en vez de
importarlas, y solo necesita pandas, numpy y openpyxl.

### El emparejamiento pregunta↔ground-truth por igualdad exacta es frágil

`benchmark_modelos.py` compara el texto de la pregunta con `==`. Una tilde o un
espacio de diferencia pierde el emparejamiento silenciosamente.
`evaluar_resultados.py` normaliza (minúsculas, sin acentos, sin puntuación) y
cae a coincidencia difusa con umbral 0.90, reportando explícitamente cuántas
consultas quedaron sin ground truth en vez de contarlas como cero.

---

## 5. Gotchas de las etapas de extracción

Recogidos al construir los pipelines por fenómeno. Los tres scripts
`src/f{1,2,3}_limpieza_extraccion.py` son **autocontenidos a propósito**
(duplican sus clases en vez de compartir un módulo); mantener esa convención.

- **Invariante de trazabilidad**: los pipelines son dirigidos por el índice
  Excel, así que **cada fila despachada debe terminar en `f{N}_documentos.jsonl`
  o en `..._pendientes.jsonl` antes de cualquier `continue`**. Un `continue`
  desnudo rompe el conteo y el hueco solo se ve en stdout efímero. Pasó con los
  catálogos de F2: 479 filas del índice pero solo 477 trazables.
- **`pytesseract` falla sobre un objeto PIL decodificado de AVIF** con
  "Unsupported image format/type", aunque Pillow lo abra bien. Solución:
  re-codificar a PNG en memoria (`io.BytesIO`) antes de pasarlo a OCR.
- **`Tipo == 'Otro'` significa cosas distintas según el fenómeno**: en F2 es una
  foto `.avif` mal etiquetada; en F3 son teselas vectoriales Mapbox `.pbf`. No
  reutilizar la regla de despacho entre fenómenos sin verificar.
- **Punkt no sirve para filas tabulares**: trata un punto tras un número como
  parte del número, así que un CSV entero se tokeniza como *una* oración
  gigante. El chunking usa el salto de línea como frontera para formatos
  tabulares.
- **`tags` cae a `topics`** en los JSON de CSIS/SWF, que no usan `tags`.
- Hay documentos en idiomas fuera de es/en/pt (traducciones oficiales de ONU,
  SWF y SIPRI). Se excluyen por lista explícita de `doc_id` verificados, **no**
  por `idioma_detectado`: langdetect da falsos positivos frecuentes en texto
  corto o técnico, y en contenido no lingüístico como los `.pbf`.

---

## 6. Lección de proceso

Antes de implementar una etapa, **releer las secciones correspondientes del
`CODEFEST_2026-1.pdf`**, no trabajar desde resúmenes. Al hacerlo para el
generador aparecieron cuatro desviaciones en un plan que estaba a punto de
ejecutarse:

1. **§8.7** exige que los post-filtros operen *"directamente sobre Metadata …
   Vectores"* → deduplicar comparando cadenas de texto no cabe; hay que hacerlo
   por similitud coseno entre vectores.
2. **§9.2.1** autoriza concatenar con *"el fragmento inmediatamente anterior o
   posterior"* — **un** vecino, en singular.
3. **§8.6** nombra max pooling / suma / media ponderada como estrategias
   habituales → una fórmula variante es admisible pero hay que justificarla.
4. **§8.6 paso 1** distingue `k_chunk` (pool de agregación) del k de búsqueda:
   son parámetros separados.

También hay una discrepancia entre el spec y los datos: **§10.1 dice que las 50
consultas están en español, inglés y portugués, pero las del archivo oficial
`data/raw/Extracto_Preguntas_50_v2.pdf` están todas en español.** Por eso el
generador lee el archivo de consultas en vez de hardcodearlas: si el jurado usa
otra versión, no se rompe.

---

## 7. Próximos pasos, por orden de rendimiento esperado

1. **Anotar más consultas en el ground truth**, sobre todo las 16 de F2. Sin
   esto no se puede validar ninguna mejora (§3.1).
2. **Rediseñar la agregación chunk→documento** con una fórmula invariante a la
   escala (candidato: rank fusion estilo RRF sobre los chunks de cada
   documento, acotando cuántos aporta cada uno). Es donde está la pérdida
   medible (§3.2), pero requiere el punto 1 para validarse.
3. **`informe_tecnico.pdf`**: estrategia de chunking, encoder y criterios de
   elección, tipo de índice FAISS, y la descripción correcta de la fórmula de
   agregación (§3.3).
4. **Grafo de conocimiento** (bonus §7), solo si sobra tiempo.
