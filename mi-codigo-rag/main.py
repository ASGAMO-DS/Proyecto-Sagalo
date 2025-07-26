import functions_framework
import json
from google.cloud import aiplatform, documentai, storage
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import os
import google.cloud.logging
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from more_itertools import chunked # MEJORA: Importado para procesar en lotes
from google.cloud.aiplatform.language_models import TextEmbeddingModel
import sys

# --- CONFIGURACIÓN (¡LLENA ESTOS VALORES!) ---
PROJECT_ID = "gen-lang-client-0054484033"
REGION = "us-central1"
QUARANTINE_BUCKET_NAME = "base-de-conocimiento-sagalo-cuarentena"
DOCUMENT_AI_PROCESSOR_ID = "f0ecc309304c8ba"
DOCUMENT_AI_PROCESSOR_VERSION_ID = "pretrained-ocr-v2.0-2023-06-02" 
VECTOR_SEARCH_INDEX_ID = "2293337163959369728"
VECTOR_SEARCH_ENDPOINT_ID = "563620655514255360"
VECTOR_SEARCH_DEPLOYED_INDEX_ID = "extremo-sagalo-v1"

print("DEBUG: El script ha iniciado.")

# --- Inicialización del Logging Profesional ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

# --- Inicialización de Clientes ---

try:
    print("DEBUG: Creando storage_client...")
    storage_client = storage.Client()
    print("DEBUG: OK - storage_client CREADO.")

    print("DEBUG: Inicializando aiplatform...")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    print("DEBUG: OK - aiplatform INICIALIZADO.")

    print("DEBUG: Creando docai_client...")
    docai_client = documentai.DocumentProcessorServiceClient(
        client_options={"api_endpoint": f"{REGION}-documentai.googleapis.com"}
    )
    print("DEBUG: OK - docai_client CREADO.")

    print("DEBUG: Creando index_endpoint...")
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=VECTOR_SEARCH_ENDPOINT_ID
    )
    print("DEBUG: OK - index_endpoint CREADO.")

    print("DEBUG: Creando embedding_model...")
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    print("DEBUG: OK - embedding_model CREADO.")

    print("DEBUG: TODAS LAS INICIALIZACIONES FUERON EXITOSAS.")

except Exception as e:
    print(f"ERROR FATAL DURANTE LA INICIALIZACIÓN: {e}", file=sys.stderr)
    # Salimos con un código de error para asegurar que el contenedor se marque como fallido
    sys.exit(1)

@functions_framework.cloud_event
def process_document_event(cloud_event):
    """Función principal que se activa con eventos de Cloud Storage."""
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    event_type = cloud_event.get("type")

    if "processed_pdfs/" in file_name or file_name.endswith('/'):
        logging.info(f"Ignorando archivo en carpeta de procesados o es una carpeta: {file_name}")
        return

    logging.info(f"Evento detectado: {event_type} para el archivo {file_name}")

    if event_type == "google.cloud.storage.object.v1.finalized":
        logging.info(f"Iniciando ingesta/actualización para {file_name}...")
        process_ingest_update(bucket_name, file_name)
    elif event_type == "google.cloud.storage.object.v1.deleted":
        logging.warning(f"Iniciando eliminación de vector para {file_name}...")
        delete_vectors_for_file(file_name)
    else:
        logging.warning(f"Tipo de evento no manejado: {event_type}")

def process_ingest_update(bucket_name, file_name):
    """Procesa un archivo, lo ingesta y lo indexa en Vector Search."""
    gcs_uri_for_docai = f"gs://{bucket_name}/{file_name}"
    
    try:
        # --- LÓGICA DE TIPO DE ARCHIVO MEJORADA ---
        original_extension = os.path.splitext(file_name)[1].lower()
        if original_extension == '.pdf':
            mime_type = "application/pdf"
        elif original_extension in ['.doc', '.docx']:
            # CORRECCIÓN: La conversión de DOCX a PDF es compleja en Cloud Functions
            # y requiere dependencias como LibreOffice. Por ahora, moveremos estos
            # archivos a cuarentena para revisión manual.
            logging.error(f"Tipo de archivo '{original_extension}' requiere conversión manual. Moviendo a cuarentena.")
            move_to_quarantine(bucket_name, file_name, "needs_conversion")
            return
        else:
            logging.error(f"Tipo de archivo no soportado: {original_extension}. Moviendo a cuarentena.")
            move_to_quarantine(bucket_name, file_name, "unsupported_type")
            return

        logging.info(f"Extrayendo texto de '{gcs_uri_for_docai}' con Document AI...")
        extracted_text = extract_text_with_document_ai(gcs_uri_for_docai, mime_type)
        if not extracted_text:
            raise ValueError("La extracción de texto con Document AI no devolvió contenido.")

        logging.info("Limpiando texto con Gemini...")
        cleaned_text = clean_text_with_gemini(extracted_text)
        
        logging.info("Extrayendo metadatos con Gemini...")
        metadata = extract_metadata_with_gemini(cleaned_text)
        
        #logging.info("Dividiendo en chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(cleaned_text)
        
        logging.info(f"Generando embeddings para {len(chunks)} chunks...")
        embeddings_to_upsert = []
        
        # --- MEJORA: Procesamiento de embeddings en lotes (batches) ---
        BATCH_SIZE = 250 
        for batch in chunked(enumerate(chunks), BATCH_SIZE):
            chunk_indices, chunk_texts = zip(*batch)
            embedding_responses = embedding_model.get_embeddings(list(chunk_texts))
            
            for i, response in enumerate(embedding_responses):
                chunk_index = chunk_indices[i]
                chunk_id = f"{hashlib.sha1(file_name.encode()).hexdigest()}_{chunk_index}"
                vector = response.values
                
                chunk_metadata = metadata.copy()
                chunk_metadata["source_file"] = file_name
                chunk_metadata["chunk_number"] = chunk_index
                
                restricts = [
                    {"namespace": key, "allow_list": [str(v)]} 
                    for key, value in chunk_metadata.items() if value and not isinstance(value, list)
                ] + [
                    {"namespace": key, "allow_list": [str(v) for v in value]}
                    for key, value in chunk_metadata.items() if isinstance(value, list) and value
                ]

                embeddings_to_upsert.append({"id": chunk_id, "embedding": vector, "restricts": restricts})

        # --- CORRECCIÓN: La llamada a `upsert` se hace UNA SOLA VEZ fuera del bucle ---
        if embeddings_to_upsert:
            logging.info(f"Indexando {len(embeddings_to_upsert)} datapoints en Vertex AI Vector Search...")
            index_endpoint.upsert_datapoints(datapoints=embeddings_to_upsert)
            logging.info(f"Archivo '{file_name}' procesado e indexado con éxito.")
        else:
            logging.warning(f"No se generaron embeddings para el archivo '{file_name}'.")

    except Exception as e:
        logging.error(f"ERROR CRÍTICO al procesar '{file_name}': {e}", exc_info=True)
        move_to_quarantine(bucket_name, file_name, "processing_error")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_text_with_document_ai(gcs_uri, mime_type):
    """Extrae texto de un documento usando una versión específica del procesador."""
    resource_name = docai_client.processor_version_path(
        PROJECT_ID, REGION, DOCUMENT_AI_PROCESSOR_ID, DOCUMENT_AI_PROCESSOR_VERSION_ID
    )
    logging.info(f"Usando la versión del procesador: {resource_name}")
    gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=mime_type)
    request = documentai.ProcessRequest(name=resource_name, gcs_document=gcs_document)
    result = docai_client.process_document(request=request)
    return result.document.text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def clean_text_with_gemini(text):
    """Limpia el texto extraído usando Gemini."""
    # --- CORRECCIÓN: Usando un nombre de modelo válido ---
    model = aiplatform.GenerativeModel("gemini-2.0-flash-lite-001")
    # Usar un límite de caracteres para evitar exceder el tamaño del prompt
    prompt = f"Limpia el siguiente texto legal, eliminando encabezados, pies de página o texto repetitivo que no sea relevante para el contenido principal. Devuelve únicamente el texto limpio y bien formado:\n---\n{text[:40000]}\n---"
    response = model.generate_content(prompt)
    return response.text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_metadata_with_gemini(text):
    """Extrae metadatos en formato JSON usando Gemini."""
    # --- CORRECCIÓN: Usando un nombre de modelo válido ---
    model = aiplatform.GenerativeModel("gemini-2.0-flash-lite-001")
    prompt = f"""Analiza el siguiente texto y extrae metadatos relevantes. Si encuentras fechas, un título, autores o un tipo de documento (ej: 'contrato', 'sentencia', 'informe'), extráelos. Responde ÚNICAMENTE con un objeto JSON. Si no encuentras nada, devuelve un JSON vacío {{}}.
Ejemplo de formato: {{"titulo": "Título del Documento", "fecha": "YYYY-MM-DD", "tipo_documento": "Contrato"}}
---
Texto:
{text[:20000]}
---
JSON de Metadatos:"""
    response = model.generate_content(prompt)
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError):
        logging.warning(f"Gemini no devolvió un JSON válido. Respuesta: {response.text}")
        return {}

def move_to_quarantine(bucket_name, file_name, error_type):
    """Mueve un archivo a un bucket de cuarentena para revisión."""
    try:
        source_bucket = storage_client.bucket(bucket_name)
        source_blob = source_bucket.blob(file_name)
        if not source_blob.exists():
            logging.warning(f"El archivo '{file_name}' ya no existe en el bucket de origen. No se puede mover a cuarentena.")
            return

        quarantine_bucket = storage_client.bucket(QUARANTINE_BUCKET_NAME)
        destination_blob_name = f"{error_type}/{os.path.basename(file_name)}"
        
        source_bucket.copy_blob(source_blob, quarantine_bucket, destination_blob_name)
        source_blob.delete()
        logging.info(f"Archivo '{file_name}' movido a cuarentena en '{destination_blob_name}'.")
    except Exception as e:
        logging.error(f"ERROR al intentar mover '{file_name}' a cuarentena: {e}", exc_info=True)

def delete_vectors_for_file(file_name):
    """
    --- CORRECCIÓN: Implementación completa de la eliminación de vectores. ---
    Encuentra y elimina todos los vectores asociados con un archivo eliminado.
    """
    try:
        # Paso 1: Usar una consulta con filtro de metadatos para encontrar los IDs de los datapoints.
        # Buscamos todos los datapoints que tengan el metadato 'source_file' igual al nombre del archivo.
        logging.info(f"Buscando datapoints para eliminar asociados con el archivo: {file_name}")
        
        # El vector de consulta puede ser cualquiera, ya que el filtro es lo que importa.
        # Usamos un vector de ceros con la dimensionalidad correcta (768 para text-embedding-004).
        dummy_vector = [0.0] * 768
        
        find_response = index_endpoint.find_neighbors(
            queries=[dummy_vector],
            deployed_index_id=VECTOR_SEARCH_DEPLOYED_INDEX_ID, # Aquí sí se usa el ID del índice desplegado.
            num_neighbors=1000, # Un número alto para asegurar que encontramos todos los chunks.
            filter=[{
                "namespace": "source_file",
                "allow_list": [file_name]
            }]
        )

        # Paso 2: Extraer los IDs de la respuesta.
        if not find_response or not find_response[0]:
            logging.warning(f"No se encontraron datapoints para el archivo '{file_name}'. Nada que eliminar.")
            return

        ids_to_delete = [neighbor.datapoint.datapoint_id for neighbor in find_response[0]]
        
        if not ids_to_delete:
            logging.warning(f"No se extrajeron IDs para el archivo '{file_name}'. Nada que eliminar.")
            return

        # Paso 3: Llamar a la API para eliminar los datapoints por sus IDs.
        logging.info(f"Eliminando {len(ids_to_delete)} datapoints para el archivo '{file_name}'...")
        index_endpoint.delete_datapoints(datapoint_ids=ids_to_delete)
        logging.info(f"Vectores para '{file_name}' eliminados con éxito.")

    except Exception as e:
        logging.error(f"ERROR CRÍTICO al eliminar vectores para '{file_name}': {e}", exc_info=True)