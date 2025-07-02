# agent_api.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field, create_model
from uuid import uuid4
from typing import List, Optional,Literal, Callable
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
import requests
from langchain.tools import StructuredTool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    """Updated tool definition model supporting both API calls and Python execution"""
    
    tool_name: Optional[str] = Field("Unnamed_tool", description="Unique name for the tool")
    tool_description: Optional[str] = Field("No description provided", description="Description of what the tool does")
    tool_type: Optional[str] = Field(None, description="Type of tool")
    # API-specific fields (optional for python_execution tools)
    endpoint_url: Optional[str] = Field(None, description="API endpoint URL")
    api_token: Optional[str] = Field(None, description="API authentication token")
    http_method: Optional[str] = Field("GET", description="HTTP method for API calls")
    
    # Tool parameters schema
    tool_parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Tool parameters definition")
    





    

# --- Routes ---

@app.get("/agent/{agent_id}/tools")
def list_agent_tools(agent_id: str):
    """List all tools for a specific agent"""
    logger.info(f"Listing tools for agent {agent_id}")

    try:
        # Get the agent document
        agent_results = list(agents_client.search(
            search_text="",
            filter=f"agent_id eq '{agent_id}'",
            top=1
        ))

        if not agent_results:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent_doc = agent_results[0]
        tools = agent_doc.get("tools", [])

        # Extract tool info from nested structure
        tool_list = []
        for tool in tools:
            tool_info = {
                "tool_id": tool.get("tool_id"),
                "tool_name": tool.get("tool_name"),
                "tool_description": tool.get("tool_description"),
                "endpoint_url": tool.get("endpoint_url"),
                "api_token": tool.get("api_token"),
                "tool_parameters": tool.get("tool_parameters"),
                "created_at": tool.get("created_at")
            }
            tool_list.append(tool_info)

        return {
            "agent_id": agent_id,
            "tools": tool_list,
            "total_tools": len(tool_list)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tools for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tools")

@app.delete("/agent/{agent_id}/tools/{tool_id}")
def delete_tool(agent_id: str, tool_id: str):
    logger.info(f"Deleting tool {tool_id} from agent {agent_id}")

    if not tool_id:
        raise HTTPException(status_code=400, detail="Tool ID cannot be empty")

    try:
        # Retrieve the agent document
        agent_docs = list(agents_client.search(
            search_text="",
            filter=f"agent_id eq '{agent_id}'",
            top=1
        ))

        if not agent_docs:
            raise HTTPException(status_code=404, detail="Agent not found")

        agent_doc = agent_docs[0]
        tools = agent_doc.get("tools", [])

        # Filter out the tool to be deleted
        updated_tools = [tool for tool in tools if tool.get("tool_id") != tool_id]

        if len(tools) == len(updated_tools):
            raise HTTPException(status_code=404, detail="Tool not found in agent")

        # Merge the updated tools back into the agent
        update_payload = {
            "agent_id": agent_id,
            "tools": updated_tools
        }

        agents_client.merge_or_upload_documents([update_payload])
        logger.info(f"Tool {tool_id} deleted from agent {agent_id}")

        return {"message": "Tool deleted successfully", "tool_id": tool_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting tool {tool_id} from agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete tool: {str(e)}")


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
def create_tools(agent_id: str, tools: List[ToolDefinition]):
    logger.info(f"Adding {len(tools)} tools to agent {agent_id}")

    # Fetch existing agent document
    results = list(agents_client.search(search_text="", filter=f"agent_id eq '{agent_id}'", top=1))
    if not results:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_doc = results[0]
    existing_tools = agent_doc.get("tools", [])

    # Prepare new tools to append
    new_tools = []
    for tool in tools:
        tool_payload = {
            "tool_id": str(uuid4()),
            "tool_name": tool.tool_name,
            "tool_description": tool.tool_description,
            "endpoint_url": tool.endpoint_url,
            "api_token": tool.api_token,
            "tool_parameters": json.dumps(tool.tool_parameters),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        new_tools.append(tool_payload)
        logger.info(f"Prepared tool {tool.tool_name} for insertion")

    # Combine existing tools and new tools
    updated_tools = existing_tools + new_tools

    # Update the agent document with the updated tools list
    updated_agent_doc = {
        "agent_id": agent_id,
        "tools": updated_tools
    }

    try:
        # Merge or upload the agent document (depending on your SDK)
        # Use merge_or_upload_documents to update partial document
        agents_client.merge_or_upload_documents(documents=[updated_agent_doc])
        logger.info(f"Successfully updated tools list for agent {agent_id}")

        return {"message": "Tools added successfully", "tools": new_tools}

    except Exception as e:
        logger.error(f"Failed to update agent document with new tools: {e}")
        raise HTTPException(status_code=500, detail="Failed to update agent tools")   

@app.post("/chat-with-tool/{agent_id}")
def chat_with_tool(agent_id: str, req: ChatRequest):
    logger.info(f"Tool-enabled chat for agent: {agent_id}")

    results = list(agents_client.search(search_text="", filter=f"agent_id eq '{agent_id}'"))
    if not results:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = results[0]

    tool_results = agent.get("tools", [])

    tools = []
    for t in tool_results:
        logger.info(f"Processing tool: {t['tool_name']}")

        input_schema_data = {}
        try:
            tool_params_str = t.get("tool_parameters", "{}")
            tool_params = json.loads(tool_params_str)
            input_schema_data = tool_params.get("input_schema", {})
        except Exception as e:
            logger.warning(f"Failed to parse tool_parameters for {t['tool_name']}: {e}")
            continue

        if not input_schema_data:
            logger.warning(f"Skipping tool {t['tool_name']} due to missing input_schema.")
            continue

        input_fields: Dict[str, tuple] = {
            k: (str, Field(..., description=v)) for k, v in input_schema_data.items()
        }
        input_schema = create_model(f"InputSchema_{t['tool_name']}", **input_fields)
        logger.info(f"Tool {t['tool_name']} schema: {input_schema.schema_json(indent=2)}")


        def build_func(schema, tool_name):
            def tool_func(**kwargs):
                logger.info(f"Tool {tool_name} called with input: {kwargs}")

                mock_weather = {
                    "New York": "Weather in New York: Sunny, 25°C",
                    "San Francisco": "Weather in San Francisco: Foggy, 18°C",
                    "London": "Weather in London: Rainy, 15°C",
                    "Paris": "Weather in Paris: Partly Cloudy, 22°C",
                    "Tokyo": "Weather in Tokyo: Mostly Sunny, 27°C",
                }
                mock_stocks = {
                    "AAPL": "$192.34",
                    "MSFT": "$342.10",
                    "GOOGL": "$139.58",
                    "TSLA": "$246.22",
                    "AMZN": "$132.44",
                }

                if "city" in kwargs:
                    city = kwargs["city"].strip()
                    logger.info(f"Resolved city input: {city}")
                    return mock_weather.get(city, f"No weather data available for '{city}'")
                elif "ticker" in kwargs:
                    ticker = kwargs["ticker"].strip().upper()
                    logger.info(f"Resolved stock ticker input: {ticker}")
                    return mock_stocks.get(ticker, f"No stock data available for '{ticker}'")
                elif "face_value" in kwargs and "years_to_maturity" in kwargs:
                    try:
                        face_value = float(kwargs["face_value"])
                        years = float(kwargs["years_to_maturity"])
                        discount_rate = 0.05  # fixed for mock
                        price = face_value / ((1 + discount_rate) ** years)
                        return f"Estimated bond price at 5% discount rate is ${price:.2f}"
                    except Exception as e:
                        return f"Error computing bond price: {str(e)}"    
                else:
                    return f"Unsupported parameter(s): {list(kwargs.keys())}"

            return tool_func

        tool_func = build_func(input_schema, t['tool_name'])

        tools.append(StructuredTool(
            name=t['tool_name'],
            description=t['tool_description'],
            func=tool_func,
            args_schema=input_schema
        ))

    if not tools:
        raise HTTPException(status_code=400, detail="No valid tools found for this agent.")

    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        deployment_name=AZURE_DEPLOYMENT,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that must use tools to answer questions. Never guess; always use a tool."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )

    try:
        result = executor.invoke({"input": req.message})

        session_id = req.session_id or str(uuid4())
        timestamp = datetime.utcnow().isoformat()
        combined_message = f"user: {req.message}\nassistant: {result['output']}"

        sessions_client.upload_documents([{
            "session_id": session_id,
            "agent_id": agent_id,
            "messages": combined_message,
            "updated_at": timestamp
        }])

        interactions = []
        agent_data = agents_client.search(
            search_text="",
            filter=f"agent_id eq '{agent_id}'",
            select=["interactions"]
        )
        for item in agent_data:
            interactions = item.get("interactions", [])
            break

        interactions.append({
            "session_id": session_id,
            "user": req.message,
            "assistant": result["output"],
            "timestamp": timestamp
        })

        agents_client.merge_or_upload_documents([{
            "agent_id": agent_id,
            "interactions": interactions
        }])

        return {"response": result["output"]}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"response": f"Sorry, I encountered an error: {str(e)}"}
        logger.error(f"Error executing agent {agent_id}: {str(e)}")
        return {"response": f"Sorry, I encountered an error: {str(e)}"}
# @app.post("/chat-with-tool/{agent_id}")
# def chat_with_tool(agent_id: str, req: ChatRequest):
#     logger.info(f"Tool-enabled chat for agent: {agent_id}")
#     logger.info(f"Received chat request for agent {agent_id}: {req}")

#     # Get the agent
#     logger.info(f"Fetching agent {agent_id}")
#     results = list(agents_client.search(search_text="", filter=f"agent_id eq '{agent_id}'"))
#     if not results:
#         logger.error(f"Agent {agent_id} not found")
#         raise HTTPException(status_code=404, detail="Agent not found")
#     agent = results[0]

#     # Fetch associated tools
#     logger.info(f"Fetching associated tools for agent {agent_id}")
#     tool_results = agents_client.search(
#         search_text="",
#         filter=f"agent_id eq '{agent_id}' and type eq 'tool'"
#     )
    
#     tools = []
#     for t in tool_results:
#         logger.info(f"Processing tool: {t['tool_name']}")
#         def create_tool_function(tool_config):
#             """Create a closure to capture the tool configuration"""
#             api_token = tool_config["api_token"]
#             url = tool_config["endpoint_url"]
#             tool_name = tool_config["tool_name"]
            
#             def tool_function(city: str) -> str:
#                 """Get weather information for a specific city"""
#                 logger.info(f"Tool {tool_name} called with city: {city}")
#                 headers = {"Authorization": f"Bearer {api_token}"}
#                 try:
#                     # For actual API calls, uncomment this:
#                     # response = requests.post(url, json={"city": city}, headers=headers)
#                     # return response.json()
                    
#                     # Demo hardcoded response:
#                     mock_weather = {
#                         "New York": "Weather in New York: Sunny, 25°C",
#                         "San Francisco": "Weather in San Francisco: Foggy, 18°C", 
#                         "London": "Weather in London: Rainy, 15°C",
#                         "Paris": "Weather in Paris: Partly Cloudy, 22°C",
#                         "Tokyo": "Weather in Tokyo: Mostly Sunny, 27°C",
#                     }

#                     city_lower = city.strip()
#                     response = mock_weather.get(city_lower, f"No weather data available for '{city}'")
#                     logger.info(f"Tool {tool_name} returning: {response}")
#                     return response
#                 except Exception as e:
#                     error_msg = f"Error getting weather for {city}: {str(e)}"
#                     logger.error(error_msg)
#                     return error_msg
            
#             return tool_function

#         # Create the tool function with proper closure
#         tool_func = create_tool_function(t)
        
#         # Create Tool object with explicit schema
#         from langchain.tools import StructuredTool
#         from pydantic import BaseModel, Field
        
#         # Define the input schema for the tool
#         class WeatherInput(BaseModel):
#             city: str = Field(description="The name of the city to get weather information for")
        
#         tool = StructuredTool(
#             name=t["tool_name"],
#             description=f"{t['tool_description']} - Use this tool when users ask about weather in any city.",
#             func=tool_func,
#             args_schema=WeatherInput
#         )
        
#         tools.append(tool)
#         logger.info(f"Created tool: {t['tool_name']} with description: {t['tool_description']}")

#     # Initialize OpenAI LLM
#     logger.info(f"Initializing OpenAI LLM with {len(tools)} tools for agent {agent_id}")
#     llm = AzureChatOpenAI(
#         api_key=AZURE_OPENAI_KEY,
#         azure_endpoint=AZURE_ENDPOINT,
#         api_version=AZURE_API_VERSION,
#         deployment_name=AZURE_DEPLOYMENT,
#         temperature=0.2
#     )

#     from langchain.agents import create_openai_functions_agent, AgentExecutor
#     from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

#     logger.info(f"Creating prompt for agent {agent_id}")
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are a helpful assistant who can provide weather related answers and has access to tools. 
#         You must only answer only using tools and not from knowledge.
                
#         When a user asks about weather in any city, you MUST use the available weather tool to get the information.
#         Do not refuse to provide weather information - use the tool provided to you.

#         Available tools: {tool_names}

#         Always use the appropriate tool when the user's question matches what the tool can do."""),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad")
#     ])
    
#     logger.info(f"Creating agent and executor for agent {agent_id}")
#     agent = create_openai_functions_agent(llm, tools, prompt)

#     # Create agent executor with explicit tool binding
#     agent_executor = AgentExecutor(
#         agent=agent, 
#         tools=tools, 
#         verbose=True,
#         handle_parsing_errors=True,
#         max_iterations=3,
#         early_stopping_method="generate"
#     )

#     # Add tool names to the prompt context
#     tool_names = [tool.name for tool in tools]
    
#     logger.info(f"Executing agent {agent_id} with input: {req.message}")
#     logger.info(f"Available tools: {tool_names}")
    
#     try:
#         result = agent_executor.invoke({
#             "input": req.message,
#             "tool_names": ", ".join(tool_names)
#         })
#         logger.info(f"Agent execution result: {result}")
#         #Updating Session index
#         session_id = req.session_id or str(uuid4())
#         timestamp = datetime.utcnow().isoformat()
#         combined_message = f"user: {req.message}\nassistant: {result['output']}"  
#         sessions_client.upload_documents(documents=[{
#         "session_id": session_id,
#         "agent_id": agent_id,
#         "messages": combined_message,
#         "updated_at": timestamp
#         }])
#         logger.info(f"Session updated: {session_id}") 

#         # Update agent interactions
#         # Fetch agent metadata from Azure AI Search
#         agent_data = agents_client.search(
#             search_text="",
#             filter=f"agent_id eq '{agent_id}'",
#             select=["interactions"]
#         )

#         existing_interactions = []
#         for item in agent_data:
#             existing_interactions = item.get("interactions", [])
#             break  # Only one agent_id match expected
#         existing_interactions.append({
#             "session_id": session_id,
#             "user": req.message,
#             "assistant": result["output"],
#             "timestamp": timestamp
#         })

#         agents_client.merge_or_upload_documents(documents=[{
#             "agent_id": agent_id,
#             "interactions": existing_interactions
#         }])
#         logger.info(f"Agent interactions updated: {agent_id}")
#         return {"response": result["output"]}
#     except Exception as e:
#         logger.error(f"Error executing agent {agent_id}: {str(e)}")
#         return {"response": f"Sorry, I encountered an error: {str(e)}"}

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