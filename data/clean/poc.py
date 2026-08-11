import json

# Ruta de tu archivo recién creado
archivo_jsonl = "f1_documentos.jsonl"

with open(archivo_jsonl, 'r', encoding='utf-8') as f:
    for linea in f:
        # 1. Accedes al documento individual
        documento = json.loads(linea)
        
        doc_id = documento["doc_id"]
        texto_completo = documento["texto_limpio"]
        
        # 2. Aquí aplicaremos el algoritmo de Chunking
        # fragmentos = fragmentar_texto(texto_completo)
        
        # 3. Y luego se guardarán junto con su metadata obligatoria
