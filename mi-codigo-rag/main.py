import functions_framework
import json
from google.cloud import aiplatform, documentai, storage
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import os       
import subprocess
import google.cloud.logging
import logging

# --- CONFIGURACIÓN (¡LLENA ESTOS VALORES!) ---
PROJECT_ID = "gen-lang-client-0054484033"
REGION = "us-central1"
QUARANTINE_BUCKET_NAME = "base-de-conocimiento-sagalo-cuarentena"
DOCUMENT_AI_PROCESSOR_ID = "85404cce273a8c03" # Debes crear un procesador en Document AI
VECTOR_SEARCH_INDEX_ID = "4565163482432929792"
VECTOR_SEARCH_ENDPOINT_ID = "2391556537668599808"
VECTOR_SEARCH_DEPLOYED_INDEX_ID = "despliegue_sagalo_v3"

# --- Inicialización del Logging Profesional ---
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

# --- Inicialización de Clientes ---
storage_client = storage.Client()
aiplatform.init(project=PROJECT_ID, location=REGION)
docai_client = documentai.DocumentProcessorServiceClient(
    client_options={"api_endpoint": f"{REGION}-documentai.googleapis.com"}
)
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name=VECTOR_SEARCH_ENDPOINT_ID
)

@functions_framework.cloud_event
def process_document_event(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    event_type = cloud_event["type"]

    if "processed_pdfs/" in file_name:
        logging.info(f"Ignorando archivo en la carpeta de procesados: {file_name}")
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
    """Procesa un archivo, convirtiéndolo a PDF si es necesario, con manejo de errores mejorado."""
    source_blob = storage_client.bucket(bucket_name).blob(file_name)
    
    temp_dir = "/tmp"
    base_name = hashlib.sha1(file_name.encode()).hexdigest()
    original_extension = os.path.splitext(file_name)[1].lower()
    download_path = os.path.join(temp_dir, f"{base_name}{original_extension}")
    pdf_path_for_processing = ""
    
    try:
        source_blob.download_to_filename(download_path)
        logging.info(f"Archivo original '{file_name}' descargado a '{download_path}'")

        if original_extension in ['.doc', '.docx']:
            logging.info(f"Iniciando conversión de '{file_name}' a PDF...")
            process = subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, download_path],
                timeout=300, capture_output=True, text=True
            )

            if process.returncode != 0:
                logging.error(f"ERROR de LibreOffice al convertir '{file_name}'. STDOUT: {process.stdout} STDERR: {process.stderr}")
                raise RuntimeError("Falla en la conversión de LibreOffice")

            converted_pdf_name = f"{base_name}.pdf"
            pdf_path_for_processing = os.path.join(temp_dir, converted_pdf_name)
            
            if not os.path.exists(pdf_path_for_processing):
                 raise FileNotFoundError(f"LibreOffice no generó el archivo PDF esperado en '{pdf_path_for_processing}'")

            logging.info(f"Archivo convertido a '{pdf_path_for_processing}'")
            
            converted_blob_name = f"processed_pdfs/{os.path.basename(file_name)}.pdf"
            converted_blob = storage_client.bucket(bucket_name).blob(converted_blob_name)
            converted_blob.upload_from_filename(pdf_path_for_processing)
            gcs_uri_for_docai = f"gs://{bucket_name}/{converted_blob_name}"
        
        elif original_extension == '.pdf':
            logging.info("Detectado archivo PDF. No se necesita conversión.")
            gcs_uri_for_docai = f"gs://{bucket_name}/{file_name}"
        else:
            raise ValueError(f"Tipo de archivo no soportado: {original_extension}")

        logging.info(f"Extrayendo texto de '{gcs_uri_for_docai}' con Document AI...")
        extracted_text = extract_text_with_document_ai(gcs_uri_for_docai)
        if not extracted_text:
            raise ValueError("La extracción de texto no devolvió contenido.")

        # --- El resto del proceso continúa exactamente igual ---
        logging.info("Limpiando boilerplate con Gemini...")
        cleaned_text = clean_text_with_gemini(extracted_text)
        logging.info("Extrayendo metadatos con Gemini...")
        metadata = extract_metadata_with_gemini(cleaned_text)
        logging.info("Dividiendo en chunks...")
        chunks = split_text_into_chunks(cleaned_text, chunk_size=1000, chunk_overlap=200)
        logging.info(f"Generando embeddings para {len(chunks)} chunks...")
        embeddings_to_upsert = []
        embedding_model = aiplatform.TextEmbeddingModel.from_pretrained("text-embedding-004")
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{hashlib.sha1(file_name.encode()).hexdigest()}_{i}"
            response = embedding_model.get_embeddings([chunk_text])
            vector = response[0].values
            chunk_metadata = metadata.copy()
            chunk_metadata["source_file"] = file_name
            chunk_metadata["chunk_number"] = i
            restricts = []
            for key, value in chunk_metadata.items():
                if value:
                    values_to_add = value if isinstance(value, list) else [value]
                    restricts.append({"namespace": key, "allow_list": [str(v) for v in values_to_add]})
            embeddings_to_upsert.append({"id": chunk_id, "embedding": vector, "restricts": restricts})
        logging.info("Indexando en Vertex AI Vector Search...")
        index_endpoint.upsert_datapoints(datapoints=embeddings_to_upsert)
        logging.info(f"Archivo '{file_name}' procesado e indexado con éxito.")

    except Exception as e:
        logging.error(f"ERROR CRÍTICO al procesar '{file_name}': {e}", exc_info=True)
        move_to_quarantine(bucket_name, file_name, "processing_error")
    
    finally:
        if os.path.exists(download_path):
            os.remove(download_path)
        pdf_to_clean = os.path.join(temp_dir, f"{base_name}.pdf")
        if os.path.exists(pdf_to_clean):
            os.remove(pdf_to_clean)
        logging.info("Limpieza de archivos temporales completada.")
        
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_text_with_document_ai(gcs_uri):
    """Extrae texto de un documento PDF usando Document AI."""
    resource_name = docai_client.processor_path(PROJECT_ID, REGION, DOCUMENT_AI_PROCESSOR_ID)
    gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=resource_name, gcs_document=gcs_document)
    result = docai_client.process_document(request=request)
    return result.document.text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def clean_text_with_gemini(text):
    model = aiplatform.GenerativeModel("gemini-1.0-pro")
    prompt = f"Limpia el siguiente texto legal, eliminando encabezados, pies de página o texto repetitivo que no sea relevante. Devuelve solo el texto limpio:\n---\n{text[:20000]}\n---"
    response = model.generate_content(prompt)
    return response.text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_metadata_with_gemini(text):
    model = aiplatform.GenerativeModel("gemini-1.0-pro")
    prompt = f"""Extrae los metadatos del siguiente documento. Responde ÚNICAMENTE con un objeto JSON.\n---\n{text[:20000]}\n---\nJSON de Metadatos:"""
    response = model.generate_content(prompt)
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        logging.warning(f"Gemini no devolvió un JSON válido para los metadatos. Respuesta: {response.text}")
        return {}

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def move_to_quarantine(bucket_name, file_name, error_type):
    try:
        source_bucket = storage_client.bucket(bucket_name)
        source_blob = source_bucket.blob(file_name)
        destination_blob_name = f"{error_type}/{os.path.basename(file_name)}"
        quarantine_bucket = storage_client.bucket(QUARANTINE_BUCKET_NAME)
        
        source_bucket.copy_blob(source_blob, quarantine_bucket, destination_blob_name)
        source_blob.delete()
        logging.info(f"Archivo '{file_name}' movido a cuarentena.")
    except Exception as e:
        logging.error(f"ERROR al intentar mover '{file_name}' a cuarentena: {e}", exc_info=True)

def delete_vectors_for_file(file_name):
    logging.warning(f"La eliminación de vectores para '{file_name}' no está completamente implementada.")