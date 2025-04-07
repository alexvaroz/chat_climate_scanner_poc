import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json



# Função para extrair texto de PDFs

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# Caminhos dos arquivos PDF

handbook_path = '.\\docs\\Handbook_POR_Jul24.pdf'
framework_path = '.\\docs\\ClimateScanner Framework_POR.pdf'



# Extração de texto e criação de lista de trechos
handbook_text = extract_text_from_pdf(handbook_path)
framework_text = extract_text_from_pdf(framework_path)
all_texts = handbook_text.split("\n\n") + framework_text.split("\n\n")



# Inicialização do modelo de embeddings

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
handbook_embeddings = embedding_model.encode(handbook_text.split("\n\n"))
framework_embeddings = embedding_model.encode(framework_text.split("\n\n"))



# Criação do índice FAISS

all_embeddings = np.vstack((handbook_embeddings, framework_embeddings))
index = faiss.IndexFlatL2(all_embeddings.shape[1])
index.add(all_embeddings)



# Função para buscar trechos relevantes

def retrieve_relevant_chunks(question, top_k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [all_texts[i] for i in indices[0]]
    return relevant_chunks



# Configuração do modelo Gemini

def get_api_key(config_file=".config.json"):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config['API_KEYS']['gemini_api_key']
    except (FileNotFoundError, KeyError):
        return None



def configure_gemini():
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="gemini-2.5-pro-exp-03-25",
                                 generation_config={"temperature": 0.7})



generative_model = configure_gemini()



# Função para gerar resposta considerando o histórico

def answer_question(question, chat_history):
    relevant_chunks = retrieve_relevant_chunks(question)

# Incluir o histórico no prompt para contextualizar

    history_text = "\n".join([f"Usuário: {msg['user']}\nAssistente: {msg['bot']}" for msg in chat_history])
    prompt = (f"Histórico da conversa:\n{history_text}\n\n"
              f"Pergunta atual: {question}\n"
              f"Informações relevantes:\n{''.join(relevant_chunks)}\n"
              f"Responda com base nas informações acima e no contexto do histórico:"
              )
    response = generative_model.generate_content(prompt)
    return response.text



# Interface com Streamlit

st.title("ChatClimateScanner")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []



# Exibir o histórico do chat
# Exibir o histórico do chat
for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(message["user"])
    with st.chat_message("assistant"):
        st.write(message["bot"])

# Campo de entrada para nova pergunta

question = st.chat_input("Digite sua pergunta aqui...")



# Processar a pergunta quando o usuário enviar

if question:

# Exibir a pergunta do usuário

    with st.chat_message("user"):
        st.write(question)
# Gerar e exibir a resposta do assistente

    with st.chat_message("assistant"):
        response = answer_question(question, st.session_state.chat_history)
        st.write(response)
# Adicionar a interação ao histórico
    st.session_state.chat_history.append({"user": question, "bot": response})