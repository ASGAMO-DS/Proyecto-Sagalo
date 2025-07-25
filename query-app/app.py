import streamlit as st
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import MatchingEngine

# --- CONFIGURACIÓN (¡LLENA ESTOS VALORES!) ---
PROJECT_ID = "gen-lang-client-0054484033"
REGION = "us-central1"
VECTOR_SEARCH_INDEX_ID = "2293337163959369728"
VECTOR_SEARCH_ENDPOINT_ID = "563620655514255360"
VECTOR_SEARCH_DEPLOYED_INDEX_ID = "extremo-sagalo-v1"


st.set_page_config(page_title="Asistente Legal IA", layout="wide")
st.title("Asistente Legal IA ⚖️")

@st.cache_resource
def get_retriever():
    """Inicializa y cachea el retriever de Vertex AI Vector Search."""
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=PROJECT_ID,
        location=REGION
    )
    
    # Usaremos uno de tus buckets existentes para el staging.
    STAGING_BUCKET = "gs://base-de-conocimiento-sagalo-cuarentena"

    vector_store = MatchingEngine.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=STAGING_BUCKET,
        embedding=embeddings,
        index_id=VECTOR_SEARCH_INDEX_ID,
        endpoint_id=VECTOR_SEARCH_ENDPOINT_ID,
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})

@st.cache_resource
def get_chain(_retriever):
    """Inicializa y cachea la cadena conversacional RAG."""
    llm = ChatVertexAI(
        # --- CORRECCIÓN 1: USAR EL NOMBRE DEL MODELO CORRECTO ---
        model_name="gemini-2.0-flash-lite-001",
        project=PROJECT_ID,
        location=REGION
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer',
        verbose=True
    )

def main():
    # Inicializar componentes
    try:
        retriever = get_retriever()
        conversation_chain = get_chain(retriever)

        # Inicializar historial de chat en session_state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Mostrar mensajes del historial
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Entrada del usuario
        if prompt := st.chat_input("Realice su consulta sobre los documentos..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Pensando..."):
                    response = conversation_chain.invoke({"question": prompt})
                    answer = response["answer"]
                    
                    source_documents = response.get("source_documents", [])
                    if source_documents:
                        sources_text = "\n\n---\n**Fuentes:**\n"
                        for doc in source_documents:
                            source_file = doc.metadata.get('source_file', 'Desconocido')
                            sources_text += f"- {source_file}\n"
                        answer += sources_text

                    message_placeholder.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Ha ocurrido un error al inicializar la aplicación: {e}")
        st.info("Por favor, verifica que todos los valores de configuración (IDs, Región, etc.) son correctos.")


if __name__ == "__main__":
    main()