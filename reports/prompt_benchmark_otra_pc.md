# Prompt para correr la comparativa de encoders en otra máquina

Copiar y pegar el bloque de abajo en Claude Code (u otro agente) en la máquina
con más cómputo. Antes de pegarlo: clonar el repo y hacer
`git checkout generador` (o hacer merge a `main` si ya se hizo el PR).

---

```
Contexto: repo laka_codefest, CODEFEST AD ASTRA 2026, Etapa 1. Necesito correr
src/benchmark_modelos.py para comparar 5 encoders candidatos (NDCG@10 + F1@3 +
Conteo de Borda) y usar el resultado real para completar la sección 4.3 de
entrega/informe_tecnico.tex (rama `generador`), que hoy dice explícitamente
que esta comparación "no se completó ni quedó persistida".

Antes de correrlo, hay tres problemas ya diagnosticados en otra máquina (GPU de
solo 4GB) que hay que resolver o verificar según el hardware disponible:

1. **`nltk.download('punkt')` cuelga sin red hacia los servidores de NLTK.**
   El script lo llama a nivel de módulo. Si pasa lo mismo aquí, bajar los
   paquetes desde el mirror de GitHub y colocarlos a mano (evita el
   downloader oficial de NLTK):
   ```bash
   mkdir -p ~/nltk_data/tokenizers
   cd /tmp
   for pkg in punkt punkt_tab; do
     curl -sL -o "${pkg}.zip" \
       "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/${pkg}.zip"
     python3 -c "import zipfile; zipfile.ZipFile('${pkg}.zip').extractall('$HOME/nltk_data/tokenizers')"
   done
   ```
   Verificar con `python3 -c "import nltk; nltk.data.find('tokenizers/punkt')"`.

2. **`batch_size=256` está hardcodeado en `ejecutar_benchmark_6_modelos`**
   (dentro de la llamada a `modelo.encode(...)` para el corpus completo). En
   una GPU de 4GB esto revienta con `CUDA out of memory` porque los textos
   llegan hasta 512 tokens. Con más VRAM probablemente no haga falta tocarlo,
   pero conviene medir antes de lanzar la corrida completa (ver punto 4).
   Si hace falta, bajarlo a 32 o 64 es seguro (medido: ~45-47 textos/s con
   `multilingual-e5-small` en una GTX 1650 de 4GB, sin diferencia real de
   throughput entre 16/32/64/96 — el cuello de botella es la longitud de
   secuencia, no el tamaño de batch).

3. **El chunking propio de `benchmark_modelos.py` (función
   `fragmentar_documento`) NO es el de producción** (`src/chunking_semantico.py`).
   Es más simple (empaquetado de oraciones por `nltk.sent_tokenize` sin
   `language=` por idioma, sin caso especial para CSV/tabular, sin fallback
   para "oraciones" que exceden el límite por sí solas). Medido: genera
   151,374 fragmentos (vs. 330,416 en producción) con una MEDIA de ~449
   palabras por chunk — muy por encima del límite de 250 que exige el spec.
   Esto es un problema real de la herramienta de benchmark, no del pipeline
   de producción: sirve para comparar encoders entre sí de forma relativa
   (los 5 modelos corren sobre el mismo chunking, así que el sesgo es
   compartido), pero **no** es representativo de la calidad del índice real
   que se entrega. Documentar esto en el informe si se usan estos números.

4. **Antes de la corrida completa, calibrar tiempo esperado.** Con 151,374
   fragmentos, medir el throughput real de cada uno de los 5 modelos sobre
   una muestra (300-1000 textos reales del corpus, no sintéticos, porque la
   longitud real importa) y extrapolar el tiempo total. En la máquina de 4GB
   e5-small dio ~46 textos/s (~55 min de sola codificación para el corpus
   completo); los modelos más grandes (`BAAI/bge-m3`, 568M parámetros;
   `paraphrase-multilingual-mpnet-base-v2` y `multilingual-e5-base`, ~278M)
   van a ser sensiblemente más lentos. Con más VRAM y/o una GPU más rápida
   debería bajar bastante, pero conviene medir antes de comprometer horas.

5. **Bug conocido en la métrica**: `calcular_ndcg_10` dentro de
   `benchmark_modelos.py` puede devolver valores >1 (evidencia:
   1.4307 observado en otra corrida). No usar el NDCG/F1/Borda que imprime el
   propio script. En vez de eso, dejar que `ejecutar_benchmark_6_modelos`
   solo genere los 5 archivos `resultados_<modelo>.jsonl` (ya tienen el mismo
   esquema que `entrega/resultados.jsonl`: query_id/documents/fragments), y
   evaluarlos por separado con la métrica ya corregida:
   ```bash
   for f in src/resultados_*.jsonl; do
     echo "=== $f ==="
     python src/evaluar_resultados.py --resultados "$f" --detalle
   done
   ```
   Armar la tabla de Conteo de Borda a mano a partir de esos 5 resultados
   (Sección 11.2 del spec: ranking por NDCG@10, ranking por F1@3, sumar
   puntos N-posición en cada tabla).

6. **Aviso importante para interpretar el resultado**: el ground truth local
   (`data/raw/FASE ORDENADA CODEFEST.xlsx`) solo cubre 7 de las 50 consultas
   (hojas F1 y F3, ninguna de F2). Ya se demostró con `entrega/generador.py`
   que con esas 7 consultas un barrido de 8 configuraciones dio NDCG@10
   IDÉNTICO en las ocho (0.1429 = 1/7 exacto) y el F1@3 solo cambiaba por una
   consulta. Es decir: **es muy probable que la comparación de los 5 modelos
   caiga en el mismo problema de falta de poder discriminativo.** Correr el
   benchmark completo igual tiene valor (deja constancia real en vez de un
   hueco), pero el resultado hay que reportarlo con la misma honestidad que
   ya tiene el resto del informe: si el ranking de Borda queda decidido por
   1-2 consultas, decirlo explícitamente en vez de presentarlo como un
   veredicto sólido.

Tareas concretas:

a. Resolver los puntos 1-2 según hardware disponible.
b. Ejecutar `ejecutar_benchmark_6_modelos` (o una versión adaptada que solo
   escriba los 5 `resultados_<modelo>.jsonl` sin depender de su propio
   cálculo de métricas) para los 5 candidatos:
   - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   - intfloat/multilingual-e5-small
   - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
   - BAAI/bge-m3
   - intfloat/multilingual-e5-base
c. Evaluar los 5 archivos de salida con `src/evaluar_resultados.py`
   (métrica ya corregida, no la del script original).
d. Armar la tabla de Conteo de Borda (Sección 11.2) a partir de esos 5
   resultados.
e. Actualizar la sección "4.3 Comparación empírica" de
   `entrega/informe_tecnico.tex` con la tabla real y sus cifras, reemplazando
   la "Nota de honestidad" actual (que dice que la corrida no se completó) por
   los resultados reales, con las salvedades del punto 6 si aplican. Recompilar
   con `tectonic entrega/informe_tecnico.tex` (o instalar tectonic sin sudo:
   `curl -sL <url-release-linux-x86_64-gnu> | tar xz` a `~/.local/bin/`, ver
   `entrega/README.md` para la referencia exacta) y confirmar que sigue en
   ≤8 páginas.
f. Commit + push a la rama `generador` (o `main` si ya se mergeó), sin
   trailer de co-autor en el mensaje (así están todos los commits anteriores
   de este repo).

Documentación de referencia ya escrita en el repo (leer antes de empezar):
`entrega/README.md` y `reports/bitacora_proyecto.md` (rama `generador`).
```

---

## Notas para retomar en esta máquina

- Ya quedaron instalados y verificados en esta máquina: `faiss-cpu`,
  `sentence-transformers`, `torch`, `nltk` (con `punkt`/`punkt_tab` bajados
  manualmente del mirror de GitHub por el bloqueo de red hacia NLTK), y
  `tectonic` en `~/.local/bin/` (motor LaTeX sin necesidad de root).
- La calibración de throughput se alcanzó a medir solo para
  `multilingual-e5-small`: ~46 textos/s estable entre `batch_size` 16/32/64/96
  (GTX 1650, 4GB VRAM), sin OOM a partir de `batch_size=32`. Con eso, el
  corpus completo de `benchmark_modelos.py` (151,374 fragmentos, chunking
  propio del script, no el de producción) tomaría ~55 min solo para ese
  modelo; los otros 4 no se llegaron a medir.
- Los 5 modelos aún no se descargaron completos en esta máquina (la
  calibración se detuvo apenas terminando el primero).
