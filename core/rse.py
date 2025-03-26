# dsrag_minimal/core/rse.py
# (Sin cambios significativos en la lógica central, solo limpieza y comentarios)
import numpy as np
from typing import List, Dict, Tuple, Any

# --- Constantes y Presets ---
# Presets para parámetros RSE (Relevant Segment Extraction)
RSE_PARAMS_PRESETS = {
    "balanced": {
        'max_length': 15,
        'overall_max_length': 30,
        'minimum_value': 0.5,
        'irrelevant_chunk_penalty': 0.18,
        'overall_max_length_extension': 5,
        'decay_rate': 30,
        'top_k_for_document_selection': 10,
        'chunk_length_adjustment': True,
    },
    "precision": { # Favorece segmentos más cortos y relevantes
        'max_length': 10, # Más corto
        'overall_max_length': 25, # Más corto
        'minimum_value': 0.6, # Más estricto
        'irrelevant_chunk_penalty': 0.25, # Penalidad más alta
        'overall_max_length_extension': 3,
        'decay_rate': 25,
        'top_k_for_document_selection': 8,
        'chunk_length_adjustment': True,
    },
    "find_all": { # Más exhaustivo, puede ser más lento y menos preciso
        'max_length': 40,
        'overall_max_length': 200,
        'minimum_value': 0.3, # Menos estricto
        'irrelevant_chunk_penalty': 0.15, # Penalidad más baja
        'overall_max_length_extension': 0,
        'decay_rate': 50, # Decaimiento más lento
        'top_k_for_document_selection': 50, # Considera más documentos
        'chunk_length_adjustment': True,
    },
    # Añadir más presets si es necesario
}

# --- Funciones Principales de RSE ---

def get_best_segments(
    all_relevance_values: List[List[float]],
    document_splits: List[int],
    max_length: int,
    overall_max_length: int,
    minimum_value: float
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Encuentra los mejores segmentos de texto basados en los valores de relevancia calculados.

    Utiliza un algoritmo greedy que itera a través de las consultas para seleccionar
    el segmento con mayor puntuación que no solape con los ya seleccionados,
    respetando las longitudes máximas y el umbral mínimo.

    Args:
        all_relevance_values: Lista de listas de scores de relevancia por chunk para cada consulta.
        document_splits: Índices que marcan el final (exclusivo) de cada documento en el meta-documento.
        max_length: Longitud máxima de un segmento individual (en chunks).
        overall_max_length: Longitud máxima combinada de todos los segmentos seleccionados.
        minimum_value: Puntuación mínima requerida para que un segmento sea considerado.

    Returns:
        Una tupla conteniendo:
            - best_segments: Lista de tuplas (inicio, fin_exclusivo) de los índices de los segmentos en el meta-documento.
            - scores: Lista de las puntuaciones correspondientes a cada segmento seleccionado.
    """
    best_segments = []
    scores = []
    total_length = 0
    num_queries = len(all_relevance_values)
    # Índices de las queries que ya no aportan segmentos válidos
    exhausted_query_indices = set()
    query_idx = 0

    # Continúa mientras no se alcance la longitud máxima total y queden queries por revisar
    while total_length < overall_max_length and len(exhausted_query_indices) < num_queries:
        current_query_idx = query_idx % num_queries # Cicla a través de las queries

        # Salta queries ya agotadas
        if current_query_idx in exhausted_query_indices:
            query_idx += 1
            continue

        relevance_values = all_relevance_values[current_query_idx]
        meta_doc_len = len(relevance_values)
        best_segment_for_query = None
        best_value_for_query = -float('inf') # Inicializa con valor muy bajo

        # Busca el mejor segmento posible para la query actual
        for start in range(meta_doc_len):
            # Optimización: Salta si el chunk inicial ya es irrelevante (score < 0)
            if relevance_values[start] < 0:
                continue

            current_sum = 0
            # Itera posibles finales para el segmento actual
            for end in range(start + 1, min(start + max_length + 1, meta_doc_len + 1)):
                # Optimización: Salta si el último chunk añadido es irrelevante
                if relevance_values[end-1] < 0 and end > start + 1: # Permite chunks iniciales negativos si son únicos
                    continue

                # --- Validaciones del Segmento ---
                segment_len = end - start
                # 1. Solapamiento con segmentos ya seleccionados
                overlaps_existing = any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments)
                if overlaps_existing: continue
                # 2. Solapamiento con límites de documentos (no cruzar documentos)
                crosses_document = any(start < split <= end for split in document_splits) # <= end porque split es exclusivo
                if crosses_document: continue
                # 3. Límite de longitud total
                if total_length + segment_len > overall_max_length: continue

                # --- Cálculo del Score (Suma simple en esta versión) ---
                # Nota: Se podría usar un cálculo más complejo si es necesario
                current_sum = sum(relevance_values[start:end])

                # Actualiza el mejor segmento para esta query si es mejor
                if current_sum > best_value_for_query:
                    best_value_for_query = current_sum
                    best_segment_for_query = (start, end)

        # Evalúa el mejor segmento encontrado para la query actual
        if best_segment_for_query and best_value_for_query >= minimum_value:
            # Añade el segmento a la lista final
            best_segments.append(best_segment_for_query)
            scores.append(best_value_for_query)
            total_length += (best_segment_for_query[1] - best_segment_for_query[0])
        else:
            # Si no se encontró segmento válido para esta query, márcala como agotada
            exhausted_query_indices.add(current_query_idx)

        query_idx += 1 # Pasa a la siguiente query (o cicla)

    # Ordena los segmentos por score descendente antes de devolverlos
    if best_segments:
         sorted_indices = np.argsort(scores)[::-1]
         best_segments = [best_segments[i] for i in sorted_indices]
         scores = [scores[i] for i in sorted_indices]

    return best_segments, scores

def get_meta_document(
    all_ranked_results: List[List[Dict[str, Any]]],
    top_k_for_document_selection: int
) -> Tuple[List[int], Dict[str, int], List[str]]:
    """
    Construye la estructura del meta-documento a partir de los resultados rankeados.

    Selecciona los documentos más relevantes basados en el top_k especificado
    y calcula los puntos de inicio/fin de cada documento dentro de una
    representación concatenada (meta-documento).

    Args:
        all_ranked_results: Lista (por query) de listas de resultados de búsqueda rankeados.
        top_k_for_document_selection: Número de resultados top por query a considerar para la selección de documentos.

    Returns:
        Una tupla conteniendo:
            - document_splits: Lista de índices que marcan el final (exclusivo) de cada documento en el meta-documento.
            - document_start_points: Diccionario mapeando doc_id al índice de inicio de su primer chunk en el meta-documento.
            - unique_document_ids: Lista ordenada de los IDs de los documentos incluidos en el meta-documento.
    """
    # 1. Recopila doc_ids de los top k resultados de cada query
    top_doc_ids_multiset = []
    for ranked_results in all_ranked_results:
        # Asegúrate de que 'metadata' y 'doc_id' existan
        ids = [r['metadata']['doc_id'] for r in ranked_results[:top_k_for_document_selection] if 'metadata' in r and 'doc_id' in r['metadata']]
        top_doc_ids_multiset.extend(ids)

    if not top_doc_ids_multiset:
        return [], {}, []

    # 2. Determina el orden de documentos basado en frecuencia y primera aparición (heurística simple)
    # Cuenta frecuencias
    from collections import Counter
    doc_id_counts = Counter(top_doc_ids_multiset)
    # Obtiene IDs únicos ordenados por frecuencia (desc) y luego por nombre (estable)
    unique_ordered_doc_ids = sorted(doc_id_counts.keys(), key=lambda x: (-doc_id_counts[x], x))

    # 3. Calcula splits y puntos de inicio
    document_splits = []
    document_start_points = {}
    current_offset = 0
    max_chunk_indices: Dict[str, int] = {} # Almacena el índice máximo de chunk por doc_id

    # Encuentra el índice máximo para cada documento seleccionado
    for results_list in all_ranked_results:
        for result in results_list:
             if 'metadata' in result:
                  doc_id = result['metadata'].get('doc_id')
                  chunk_idx = result['metadata'].get('chunk_index')
                  if doc_id in unique_ordered_doc_ids and chunk_idx is not None:
                       max_chunk_indices[doc_id] = max(max_chunk_indices.get(doc_id, -1), int(chunk_idx))

    # Construye los splits y start_points
    for doc_id in unique_ordered_doc_ids:
        document_start_points[doc_id] = current_offset
        # El tamaño del documento es max_chunk_index + 1
        doc_size = max_chunk_indices.get(doc_id, -1) + 1
        if doc_size > 0:
             current_offset += doc_size
             document_splits.append(current_offset)
        else:
             # Si un documento seleccionado no tiene chunks válidos (raro), se elimina
             del document_start_points[doc_id]
             print(f"Warning: Document '{doc_id}' selected but no valid chunk indices found. Excluding.")

    # Filtra la lista de IDs únicos para que coincida con los que realmente tienen chunks
    final_unique_doc_ids = list(document_start_points.keys())

    return document_splits, document_start_points, final_unique_doc_ids


def _get_chunk_value(rank: int, similarity_score: float, irrelevant_chunk_penalty: float, decay_rate: float) -> float:
    """
    Calcula el valor de relevancia de un chunk individual.

    Args:
        rank: Posición del chunk en los resultados rerankeados (0-based).
        similarity_score: Score de similaridad devuelto por el vector DB o reranker (idealmente 0-1).
        irrelevant_chunk_penalty: Penalización base aplicada a todos los chunks.
        decay_rate: Controla cuán rápido decae el valor con el rank.

    Returns:
        Valor de relevancia del chunk.
    """
    # Asegura que el score de similaridad esté en [0, 1]
    bounded_score = max(0.0, min(1.0, similarity_score))
    # Calcula el valor basado en el decaimiento exponencial y el score, luego resta la penalidad
    # Usamos rank + 1 para evitar división por cero si decay_rate es 0 o rank es 0 en algunas fórmulas
    # La fórmula original: np.exp(-rank / decay_rate) * absolute_relevance_value - irrelevant_chunk_penalty
    # Usamos bounded_score como absolute_relevance_value
    # Ajustamos para que decay_rate=0 no cause error y ranks más bajos den scores más altos
    if decay_rate <= 0: decay_rate = 1e-6 # Evita división por cero
    decay_factor = np.exp(-(rank + 1) / decay_rate) # +1 para que rank 0 no sea exp(0)=1 siempre

    # Combina score y decaimiento. Pondera más el score inicial.
    # Una opción simple: Usa el score directamente ponderado por el decaimiento.
    value = bounded_score * decay_factor

    # Aplica la penalización base
    return value - irrelevant_chunk_penalty

def get_relevance_values(
    all_ranked_results: List[List[Dict[str, Any]]],
    meta_document_length: int,
    document_start_points: Dict[str, int],
    unique_document_ids: List[str], # Lista de IDs en el meta-documento
    irrelevant_chunk_penalty: float,
    decay_rate: float = 30.0,
    chunk_length_adjustment: bool = True,
    reference_chunk_length: int = 700, # Longitud de referencia para ajuste
) -> List[List[float]]:
    """
    Calcula los valores de relevancia para cada chunk en el meta-documento, para cada consulta.

    Args:
        all_ranked_results: Lista (por query) de listas de resultados rerankeados.
        meta_document_length: Longitud total del meta-documento (en chunks).
        document_start_points: Diccionario mapeando doc_id a su índice de inicio en el meta-documento.
        unique_document_ids: Lista de IDs de documentos incluidos en el meta-documento.
        irrelevant_chunk_penalty: Penalización base para cada chunk.
        decay_rate: Tasa de decaimiento para el rank.
        chunk_length_adjustment: Si se ajusta el score por longitud del chunk.
        reference_chunk_length: Longitud base para el ajuste por longitud.

    Returns:
        Lista (por query) de listas de valores de relevancia para cada chunk del meta-documento.
    """
    all_relevance_values = []
    num_queries = len(all_ranked_results)

    # Pre-calcula longitudes de chunk si es necesario (solo se necesita una vez)
    chunk_lengths = np.zeros(meta_document_length, dtype=int)
    if chunk_length_adjustment:
        # Necesitamos iterar sobre todos los resultados para obtener las longitudes
        # Esto asume que los resultados contienen la longitud o el texto
        processed_indices = set() # Para evitar procesar la misma longitud múltiples veces
        for results_list in all_ranked_results:
            for result in results_list:
                 meta = result.get('metadata')
                 if meta:
                      doc_id = meta.get('doc_id')
                      chunk_idx = meta.get('chunk_index')
                      if doc_id in document_start_points and chunk_idx is not None:
                           meta_doc_idx = document_start_points[doc_id] + int(chunk_idx)
                           if meta_doc_idx < meta_document_length and meta_doc_idx not in processed_indices:
                                chunk_text = meta.get('chunk_text', '') # Asume que chunk_text está en metadata
                                chunk_lengths[meta_doc_idx] = len(chunk_text)
                                processed_indices.add(meta_doc_idx)

    # Calcula relevance_values para cada query
    for i in range(num_queries):
        ranked_results = all_ranked_results[i]
        # Inicializa valores con la penalización base negativa
        # Esto asegura que chunks no encontrados tengan score negativo
        relevance_values = np.full(meta_document_length, -irrelevant_chunk_penalty, dtype=float)

        # Asigna scores positivos basados en los resultados rankeados
        for rank, result in enumerate(ranked_results):
            meta = result.get('metadata')
            if not meta: continue

            doc_id = meta.get('doc_id')
            chunk_idx = meta.get('chunk_index')
            similarity = result.get('similarity') # Score del reranker/vectorDB

            # Verifica que el chunk pertenezca al meta-documento y tenga índice y score válidos
            if doc_id in document_start_points and chunk_idx is not None and similarity is not None:
                meta_doc_idx = document_start_points[doc_id] + int(chunk_idx)
                # Asegura que el índice esté dentro de los límites
                if 0 <= meta_doc_idx < meta_document_length:
                    # Calcula el valor base usando rank y score
                    base_value = _get_chunk_value(rank, float(similarity), irrelevant_chunk_penalty, decay_rate)

                    # Ajusta por longitud si está habilitado
                    if chunk_length_adjustment:
                         chunk_len = chunk_lengths[meta_doc_idx]
                         if chunk_len > 0 and reference_chunk_length > 0:
                              # Ajuste simple: escala por ratio de longitud, pero solo aumenta si es más largo
                              # La fórmula original era: base_value * (max(chunk_len, ref) / ref)
                              # Simplifiquemos a: base_value * (chunk_len / ref) if chunk_len > ref else base_value
                              # O la fórmula original si se prefiere:
                              length_factor = max(chunk_len, reference_chunk_length) / reference_chunk_length
                              # Podríamos querer limitar el impacto del ajuste de longitud
                              # length_factor = 1 + max(0, (chunk_len - reference_chunk_length) / reference_chunk_length * 0.5) # Ejemplo: 50% del efecto
                              base_value *= length_factor

                    # Asigna el valor calculado (sobrescribe la penalización base)
                    relevance_values[meta_doc_idx] = base_value

        all_relevance_values.append(relevance_values.tolist()) # Convierte a lista

    return all_relevance_values

# Nota: adjust_relevance_values_for_chunk_length fue integrada en get_relevance_values