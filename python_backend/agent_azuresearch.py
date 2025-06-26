# agent_api.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from uuid import uuid4
from typing import List, Optional
from datetime import datetime
import os
from pydantic import BaseModel
from typing import Dict, Any
import logging
from json import dumps
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain.schema import BaseMessage
import fitz
from azure.search.documents.models import VectorizedQuery
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema.messages import HumanMessage
from langchain_core.messages import HumanMessage as VisionHumanMessage
import base64
import time
from dotenv import load_dotenv
load_dotenv() 
# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Azure AI Search Setup ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AGENTS_INDEX = os.getenv("AGENTS_INDEX", "agents-index")
SESSIONS_INDEX = os.getenv("SESSIONS_INDEX", "sessions-index")
DOCS_INDEX = os.getenv("DOCS_INDEX", "agent-docs-index")

search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
agents_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AGENTS_INDEX, credential=search_credential)
sessions_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=SESSIONS_INDEX, credential=search_credential)
docs_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=DOCS_INDEX, credential=search_credential)

# --- Azure OpenAI Setup ---
from langchain_openai import AzureChatOpenAI
from langchain.schema import AIMessage, HumanMessage,SystemMessage
from langchain.retrievers import BM25Retriever, ContextualCompressionRetriever

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_RESOURCE_NAME = os.getenv("AZURE_RESOURCE_NAME")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_GPT4O_VISION_DEPLOYMENT = os.getenv("AZURE_GPT4O_VISION_DEPLOYMENT")
AZURE_DEPLOYMENT_EMBEDDING = os.getenv("AZURE_DEPLOYMENT_EMBEDDING")
# --- FastAPI ---
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # React's URL
    allow_credentials=True,
    allow_methods=["*"],  # or ["GET", "POST", "PATCH"] specifically
    allow_headers=["*"],
)
# --- Pydantic Models ---
class AgentCreateRequest(BaseModel):
    name: str
    model_name: str
    system_prompt: str
    temperature: float
    retriever_strategy: Optional[str] = None

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    retriever_strategy: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    doc_id: Optional[str] = None
    file_name: Optional[str] = None

class SearchDocsRequest(BaseModel):
    query: str
    top_k: int = 3

class ToolDefinition(BaseModel):
    tool_name: str
    tool_description: str
    endpoint_url: str
    api_token: str
    parameters: Dict[str, Any]  # This is the OpenAI-style parameter schema

# --- Routes ---
@app.post("/agent")
def create_agent(req: AgentCreateRequest):
    agent_id = str(uuid4())
    document = {
        "agent_id": agent_id,
        "name": req.name,
        "model_name": req.model_name,
        "system_prompt": req.system_prompt,
        "temperature": req.temperature,
        "retriever_strategy": req.retriever_strategy or "",
        "interactions": []
    }
    result = agents_client.upload_documents([document])
    logger.info(f"Agent created: {agent_id}")
    return {"agent_id": agent_id, "result": result}

@app.get("/agent/{agent_id}")
def get_agent(agent_id: str):
    logger.info(f"Getting agent: {agent_id}")
    
    # Get agent
    results = list(agents_client.search(search_text="*", filter=f"agent_id eq '{agent_id}'", include_total_count=True))
    if not results:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = results[0]
    
    # Get unique uploaded file names for the agent
    doc_results = docs_client.search(
        search_text="",
        filter=f"agent_id eq '{agent_id}'",
        select=["file_name"]
    )
    filenames = set()
    for doc in doc_results:
        fname = doc.get("file_name")
        if fname:
            filenames.add(fname)

    return {
        "agent": agent,
        "documents": list(filenames)
    }

@app.get("/agents")
def get_all_agents():
    logger.info("Fetching all agents with their documents")
    agent_results = list(agents_client.search(search_text="*"))
    agents_with_docs = []

    for agent in agent_results:
        agent_id = agent["agent_id"]
        doc_results = docs_client.search(
            search_text="",
            filter=f"agent_id eq '{agent_id}'",
            select=["file_name"]
        )
        filenames = set()
        for doc in doc_results:
            fname = doc.get("file_name")
            if fname:
                filenames.add(fname)

        agents_with_docs.append({
            "agent": {
                "agent_id": agent.get("agent_id"),
                "name": agent.get("name"),
                "model_name": agent.get("model_name"),
                "system_prompt": agent.get("system_prompt"),
                "temperature": agent.get("temperature"),
                "retriever_strategy": agent.get("retriever_strategy")
            },
            "documents": list(filenames)
        })

    return agents_with_docs

@app.post("/agent/{agent_id}/tools")
def create_tool(agent_id: str, tool: ToolDefinition):
    logger.info(f"Creating tool '{tool.tool_name}' for agent {agent_id}")

    tool_payload = {
        "tool_id": str(uuid4()),  # this becomes tool_id
        "agent_id": agent_id,
        "type": "tool",
        "tool_name": tool.tool_name,
        "tool_description": tool.tool_description,
        "endpoint_url": tool.endpoint_url,
        "api_token": tool.api_token,
        "tool_parameters": json.dumps(tool.parameters),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    try:
        agents_client.upload_documents(documents=[tool_payload])
        logger.info(f"Tool '{tool.tool_name}' created for agent {agent_id}")
        return {"message": "Tool registered successfully.", "tool": tool_payload}
    except Exception as e:
        logger.error(f"Tool creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to persist tool")

@app.post("/chat-with-tool/{agent_id}")
def chat_with_tool(agent_id: str, req: ChatRequest):
    logger.info(f"Tool-enabled chat for agent: {agent_id}")
    logger.info(f"Received chat request for agent {agent_id}: {req}")

    # Get the agent
    logger.info(f"Fetching agent {agent_id}")
    results = list(agents_client.search(search_text="", filter=f"agent_id eq '{agent_id}'"))
    if not results:
        logger.error(f"Agent {agent_id} not found")
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = results[0]

    # Fetch associated tools
    logger.info(f"Fetching associated tools for agent {agent_id}")
    tool_results = agents_client.search(
        search_text="",
        filter=f"agent_id eq '{agent_id}' and type eq 'tool'"
    )

    functions = []
    name_to_func = {}
    for t in tool_results:
        func_def = {
            "name": t["name"],
            "description": t["tool_description"],
            "parameters": t["tool_parameters"]
        }

        def call_api_wrapper(api_token=t["api_token"], url=t["endpoint_url"]):
            def inner_func(user_input: str):  # accept single positional argument
                headers = {"Authorization": f"Bearer {api_token}"}
                try:
                    # For example, send the user_input as JSON payload
                    # response = requests.post(url, json={"input": user_input}, headers=headers)
                    
                    # Demo hardcoded response:
                    response = "Huzzah!! Pirates of the Caribbean"
                    
                    # If you use requests uncomment below
                    # return response.json()
                    
                    return response
                except Exception as e:
                    return {"error": str(e)}
            return inner_func

        functions.append(func_def)
        name_to_func[t["name"]] = call_api_wrapper()

    # Initialize OpenAI LLM with tools
    logger.info(f"Initializing OpenAI LLM with tools for agent {agent_id}")
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        deployment_name=AZURE_DEPLOYMENT,
        temperature=0.2
    )

    from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
    from langchain.tools import Tool
    from langchain.agents import create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    logger.info(f"Creating prompt for agent {agent_id}")
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses the provided tools to answer questions."),
    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad")  # required!
    ])

    tools = [Tool.from_function(f, name=n, description="Tool Call") for n, f in name_to_func.items()]
    
    logger.info(f"Creating agent and executor for agent {agent_id}")
    agent = create_openai_functions_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    logger.info(f"Executing agent {agent_id} with input {req.message}")
    result = agent_executor.invoke({"input": req.message})
    logger.info(f"Returning response for agent {agent_id}: {result['output']}")
    return {"response": result["output"]}

@app.patch("/agent/{agent_id}")
def update_agent(agent_id: str, req: AgentUpdateRequest):
    logger.info(f"Updating agent: {agent_id}")
    results = list(agents_client.search(search_text="*", filter=f"agent_id eq '{agent_id}'"))
    if not results:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = results[0]
    update_data = req.dict(exclude_none=True)
    agent.update(update_data)
    agents_client.upload_documents([agent])
    logger.info(f"Agent updated: {agent_id}")
    return {"message": "Agent updated"}

@app.delete("/agent/{agent_id}")
def delete_agent(agent_id: str):
    logger.info(f"Deleting agent: {agent_id}")
    agents_client.delete_documents([{"agent_id": agent_id}])
    logger.info(f"Agent deleted: {agent_id}")
    return {"message": "Agent deleted"}

@app.post("/chat/{agent_id}")
def chat(agent_id: str, req: ChatRequest):
    logger.info(f"Chatting with agent: {agent_id} | Session ID: {req.session_id or '[NEW]'}")

    # --- Get Agent ---
    results = list(agents_client.search(search_text="", filter=f"agent_id eq '{agent_id}'"))
    if not results:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = results[0]

    session_id = req.session_id or str(uuid4())

    # --- Get or Initialize Session ---
    session_results = list(sessions_client.search(search_text="", filter=f"agent_id eq '{agent_id}' and session_id eq '{session_id}'"))
    session = session_results[0] if session_results else {}

    messages = session.get("messages", [])
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            logger.warning("Failed to parse session messages as JSON. Starting fresh.")
            messages = []

    
    messages.append({"role": "user", "content": req.message})

    # --- LLM Setup ---
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        deployment_name=AZURE_DEPLOYMENT,
        temperature=0.2
    )

    # --- Retriever ---
    retriever_strategy = agent.get("retriever_strategy")
    retrieved_docs = []
    if retriever_strategy == "bm25":
        logger.info("Using BM25 retriever")
        retriever = BM25Retriever.from_texts([m["content"] for m in messages if m["role"] == "user"])
        retrieved_docs = retriever.invoke(req.message)
    elif retriever_strategy == "contextual_compression":
        logger.info("Using Contextual Compression retriever")
        base = BM25Retriever.from_texts([m["content"] for m in messages])
        retriever = ContextualCompressionRetriever(base_compressor=llm, base_retriever=base)
        retrieved_docs = retriever.invoke(req.message)
    elif retriever_strategy == "vector_rag":
        logger.info("Using Vector RAG retriever")
        embedder = AzureOpenAIEmbeddings(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
            deployment=AZURE_DEPLOYMENT_EMBEDDING
        )
        query_vector = embedder.embed_query(req.message)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=5,
            fields="embedding"
        )
        doc_filter = f"agent_id eq '{agent_id}'"
        if req.file_name:
            doc_filter += f" and file_name eq '{req.file_name}'"
        if req.doc_id:
            doc_filter += f" and doc_id eq '{req.doc_id}'"
        
        logger.info(f"Retrieving documents with filter: {doc_filter}")
        doc_results = docs_client.search(
            search_text="",
            vector_queries=[vector_query],
            filter=doc_filter
        )
        retrieved_docs = [r["content"] for r in doc_results]
        logger.info(f"Retrieved documents: {len(retrieved_docs)}")
    # --- Inject Retrieved Context ---
    retrieved_context = ""
    if retrieved_docs:
        retrieved_context = "\n".join([
            doc.page_content if hasattr(doc, "page_content") else str(doc) 
            for doc in retrieved_docs
        ])
    base_system_prompt = agent["system_prompt"]
    logger.info(f"Base system prompt: {base_system_prompt[:200]}...")
    logger.info(f"Retrieved context: {retrieved_context[:200]}...")
    


    if retriever_strategy == "vector_rag" and retrieved_context:
        messages[-1]["content"] = f"""IMPORTANT INSTRUCTIONS:
            - You must ONLY use information from the provided context below to answer questions
            - If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer that question."
            - Do not use your general knowledge or training data
            - Stay strictly within the bounds of the provided context

            CONTEXT:
            {retrieved_context}

            QUESTION:
            
            {messages[-1]['content']}"""

    lc_messages = [
        SystemMessage(base_system_prompt),
        *[HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in messages if m["role"] != "system"]
    ]

    logger.info(f"Number of messages sent to LLM: {len(lc_messages)}")
    response = llm.invoke(lc_messages)
    logger.info(f"LLM responded: {response.content}")

    messages.append({"role": "assistant", "content": response.content})

    # --- Store Session ---
    sessions_client.upload_documents([{
        "session_id": session_id,
        "agent_id": agent_id,
        "messages": json.dumps(messages),
        "updated_at": datetime.utcnow().isoformat()
    }])
    logger.info(f"Session {session_id} updated with new message.")

    # --- Update Agent Interaction Log ---
    interactions = agent.get("interactions", [])
    interactions.append({
        "session_id": session_id,
        "user": req.message,
        "assistant": response.content,
        "timestamp": datetime.utcnow().isoformat()
    })
    agents_client.upload_documents([{
        **agent,
        "interactions": interactions
    }])

    return {"session_id": session_id, "response": response.content}


# --- Upload and Index Documents ---
@app.post("/agent/{agent_id}/upload-docs")
def upload_docs(agent_id: str, file: UploadFile = File(...)):
    logger.info(f"Uploading document for agent {agent_id} | File: {file.filename}")

    # --- Extract Text from File ---
    text = ""
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        file_bytes = file.file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                text += page_text

            if not text.strip():  # Fall back to OCR if no text was extracted
                logger.info("No extractable text found in PDF. Falling back to OCR using GPT-4o Vision.")
                vision_llm = AzureChatOpenAI(
                    api_key=AZURE_OPENAI_KEY,
                    azure_endpoint=AZURE_ENDPOINT,
                    api_version=AZURE_API_VERSION,
                    deployment_name=AZURE_GPT4O_VISION_DEPLOYMENT,
                    temperature=0.0
                )
                for page_index in range(len(doc)):
                    pix = doc.load_page(page_index).get_pixmap(dpi=200)
                    image_bytes = pix.tobytes("jpeg")
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                    message = VisionHumanMessage(content=[
                        {"type": "text", "text": "Extract all visible text from this image."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ])
                    vision_response = vision_llm.invoke([message])
                    text += vision_response.content + "\n"

    elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        try:
            image_bytes = file.file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            # Initialize GPT-4o with vision support
            vision_llm = AzureChatOpenAI(
                api_key=AZURE_OPENAI_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION,
                deployment_name=AZURE_GPT4O_VISION_DEPLOYMENT,
                temperature=0.0
            )
            image_url = f"data:image/jpeg;base64,{base64_image}"
            message = VisionHumanMessage(content=[
                {"type": "text", "text": "Extract all visible text from this image."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])
            vision_response = vision_llm.invoke([message])
            text = vision_response.content
        except Exception as e:
            logger.error(f"Azure GPT-4o Vision OCR error: {e}")
            raise HTTPException(status_code=500, detail="Failed to process image with GPT-4o Vision")

    else:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, JPEG, or PNG files are supported")

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Embed chunks
    embedder = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        deployment=AZURE_DEPLOYMENT_EMBEDDING
    )
    docs = []
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        try:
            batch_embeddings = embedder.embed_documents(batch_chunks)
        except Exception as e:
            logger.error(f"Embedding error on batch {i}-{i+batch_size}: {e}")
            continue
        for chunk, emb in zip(batch_chunks, batch_embeddings):
            docs.append({
                "doc_id": str(uuid4()),
                "agent_id": agent_id,
                "file_name": file.filename,
                "content": chunk,
                "embedding": emb,
                "upload_date": datetime.utcnow().isoformat() + "Z"
            })
        time.sleep(1)
    upload_batch_size = 10
    for i in range(0, len(docs), upload_batch_size):
        batch = docs[i:i+upload_batch_size]
        try:
            docs_client.upload_documents(documents=batch)
            logger.info(f"Uploaded batch {i}-{i+len(batch)} of {len(docs)} total chunks")
        except Exception as e:
            logger.error(f"Upload error on batch {i}-{i+len(batch)}: {e}")
        time.sleep(1)
    logger.info(f"Uploaded {len(docs)} chunks for {file.filename}")
    return {"uploaded_chunks": len(docs)}

# --- Search Documents by Vector ---
@app.post("/agent/{agent_id}/search-docs")
def search_docs(agent_id: str, req: SearchDocsRequest):
    embedder = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        deployment=AZURE_DEPLOYMENT_EMBEDDING
    )
    query_vector = embedder.embed_query(req.query)
    vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=req.top_k,
                    fields="embedding"
                        )
    results = docs_client.search(
        search_text="",
        vector_queries=[vector_query],
        filter=f"agent_id eq '{agent_id}'"
    )
    return [{
        "file_name": r["file_name"],
        "content": r["content"],
        "score": r["@search.score"],
        "upload_date": r.get("upload_date")
    } for r in results]

# --- List Documents Uploaded to Agent ---
@app.get("/agent/{agent_id}/docs")
def list_docs(agent_id: str):
    logger.info(f"Fetching unique file names and counts for agent: {agent_id}")
    
    results = docs_client.search(
        search_text="",
        filter=f"agent_id eq '{agent_id}'",
        select=["file_name"]
    )

    file_counts = {}
    for r in results:
        fname = r.get("file_name")
        if fname:
            file_counts[fname] = file_counts.get(fname, 0) + 1

    return [
        {"file_name": name, "count": count}
        for name, count in file_counts.items()
    ]

@app.get("/agent/{agent_id}/docs/by-file-name")
def list_docs_by_file_name(agent_id: str, file_name: str):
    logger.info(f"Fetching documents for agent: {agent_id} with file name: {file_name}")
    
    results = docs_client.search(
        search_text="",
        filter=f"agent_id eq '{agent_id}' and file_name eq '{file_name}'"
    )

    return [
        {
            "doc_id": r.get("doc_id"),
            "file_name": r.get("file_name"),
            "content": r.get("content"),
            "upload_date": r.get("upload_date")
        }
        for r in results
    ]

@app.get("/agent/{agent_id}/file-names")
def list_unique_filenames(agent_id: str):
    logger.info(f"Fetching unique file names for agent: {agent_id}")
    
    results = docs_client.search(
        search_text="",
        filter=f"agent_id eq '{agent_id}'",
        select=["file_name"]
    )

    filenames = set()
    for r in results:
        fname = r.get("file_name")
        if fname:
            filenames.add(fname)

    return {"file_name": list(filenames)}

@app.delete("/agent/{agent_id}/delete-document")
def delete_document(agent_id: str, file_name: str = Query(..., description="File name to delete")):
    logger.info(f"Deleting document for agent {agent_id} | File: {file_name}")

    # Search for documents with the given agent_id and file_name
    results = list(docs_client.search(
        search_text="",
        filter=f"agent_id eq '{agent_id}' and file_name eq '{file_name}'",
        select=["doc_id"]
    ))

    if not results:
        raise HTTPException(status_code=404, detail="No matching documents found")

    # Extract doc_ids to delete
    doc_ids_to_delete = [{"doc_id": doc["doc_id"]} for doc in results]

    try:
        docs_client.delete_documents(documents=doc_ids_to_delete)
        logger.info(f"Deleted {len(doc_ids_to_delete)} documents for file: {file_name}")
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete documents")

    return {"deleted_docs": len(doc_ids_to_delete)}