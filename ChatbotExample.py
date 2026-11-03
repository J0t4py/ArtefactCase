import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
MODEL_PATH = BASE_DIR / "gemma-2-2b-it-Q4_K_M.gguf"
N_CORES = 6

#Download do Modelo GGUF
if not MODEL_PATH.exists():
    print("Modelo GGUF não encontrado na pasta do projeto.")
    print("Baixando gemma-2-2b-it-Q4_K_M.gguf do Hugging Face (~1.6 GB)...")
    hf_hub_download(
        repo_id="bartowski/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        local_dir=str(BASE_DIR)
    )
    print("Download concluído!\n")

#Carregamento dos PDFs e CSVs
def carregar_documentos(diretorio: Path):
    if not diretorio.exists():
        print(f"A pasta '{diretorio}' não existe.")
        sys.exit(1)
        
    documentos = []
    
    # PDFs
    pdf_loader = DirectoryLoader(
        str(diretorio),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs_pdf = pdf_loader.load()
    documentos.extend(docs_pdf)
    print(f"-> {len(docs_pdf)} página(s) de PDF carregada(s).")
    
    # CSVs
    csv_files = list(diretorio.glob("**/*.csv"))
    for csv_file in csv_files:
        csv_loader = CSVLoader(str(csv_file), encoding="utf-8")
        docs_csv = csv_loader.load()
        documentos.extend(docs_csv)
    print(f"-> {len(csv_files)} arquivo(s) CSV processado(s).")
    
    return documentos

#Inicialização do Banco de dados
def inicializar_vector_store():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("Carregando banco existente em chroma_db/...")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )
    
    print(f"Indexando arquivos '{DATA_DIR.relative_to(BASE_DIR)}'...")
    documentos = carregar_documentos(DATA_DIR)
    
    if not documentos:
        print("Nenhum arquivo PDF ou CSV encontrado.")
        sys.exit(1)
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documentos)
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    return vector_store

#Inicialização do Modelo Gemma 2
def inicializar_llm():
    return LlamaCpp(
        model_path=str(MODEL_PATH),
        temperature=0.1,
        max_tokens=256,              # Limite reduzido para respostas rápidas
        n_ctx=1024,                  # Janela de contexto ideal para RAG sem sobrecarregar a CPU
        n_threads=N_CORES,           # Paralelismo na geração do texto (6 threads)
        n_threads_batch=N_CORES,     # Paralelismo no processamento do prompt (6 threads)
        n_batch=512,                 # Lote de tokens processados por ciclo de CPU
        f16_kv=True,                 # Cache KV otimizado
        verbose=False
    )

#Pipeline do RAG e Chat no Terminal
def main():
    vector_store = inicializar_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = inicializar_llm()
    
    print("\n" + "="*50)
    print(f"   CHATBOT RAG LOCAL - GEMMA 2 (CPU: {N_CORES} Cores)")
    print("   Digite 'sair' para encerrar a sessão.")
    print("="*50 + "\n")
    
    while True:
        try:
            pergunta = input("\nVocê: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in ["sair", "exit", "quit"]:
                print("Encerrando chatbot. Até logo!")
                break
                
            docs_relevantes = retriever.invoke(pergunta)
            contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
            
            prompt = (
                f"<start_of_turn>user\n"
                f"Você é um assistente virtual (chatbot) da loja de instrumentos Empório da música em Campo grande (Rua 14 de maio 3200) preciso. Responda à pergunta usando APENAS o contexto fornecido.\n\n"
                f"CONTEXTO:\n{contexto}\n\n"
                f"PERGUNTA: {pergunta}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            
            print("\nAssistente: ", end="", flush=True)
            resposta = llm.invoke(prompt)
            print(resposta)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()