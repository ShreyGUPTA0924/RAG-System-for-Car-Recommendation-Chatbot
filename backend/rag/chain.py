"""
ConversationalRetrievalChain setup for RAG chatbot.
Compatible with LangChain 1.0+ using LCEL pattern.
"""

import logging
import inspect
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# LangChain 1.0+ imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from backend.rag.model import get_llm
from backend.rag.retriever import get_retriever

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load prompt template
PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "base_prompt.txt"


def load_prompt_template() -> str:
    """Load prompt template from file."""
    if PROMPT_TEMPLATE_PATH.exists():
        with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Default prompt if file doesn't exist
        return """You are a helpful car recommendation assistant. Use the following pieces of context to answer the user's question about cars.

Context:
{context}

Question: {question}

Instructions:
- Use ONLY the information provided in the context above
- Cite specific fields and values from the context (e.g., "The Toyota Camry Hybrid has a price of $28,000")
- If the context doesn't contain enough information to answer the question, say so explicitly
- Do not make up or hallucinate any information
- Provide clear, concise recommendations based on the context

Answer:"""


def _call_sync_or_async(func, *args, **kwargs):
    """Call sync or async function from sync code (block on coroutine)."""
    if inspect.iscoroutinefunction(func):
        # If an event loop is already running (e.g., in some async contexts), use asyncio.run
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run(func(*args, **kwargs))
            else:
                return loop.run_until_complete(func(*args, **kwargs))
        except RuntimeError:
            return asyncio.run(func(*args, **kwargs))
    else:
        return func(*args, **kwargs)


def _robust_retrieve(retriever, query):
    """
    Try several retrieval method names to be compatible with different LangChain versions.
    Returns a list of Document-like objects.
    """
    # LangChain 1.0+ uses invoke method (Runnable interface)
    if hasattr(retriever, "invoke"):
        try:
            return retriever.invoke(query)
        except Exception as e:
            # Fallback to other methods if invoke fails
            pass
    
    # Try public method first
    if hasattr(retriever, "get_relevant_documents"):
        try:
            return _call_sync_or_async(retriever.get_relevant_documents, query)
        except TypeError:
            # If it requires run_manager, try invoke instead
            if hasattr(retriever, "invoke"):
                return retriever.invoke(query)
    
    # Try retrieve method
    if hasattr(retriever, "retrieve"):
        return _call_sync_or_async(retriever.retrieve, query)
    
    # Last resort: try invoke with proper format
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    
    raise RuntimeError(
        "Retriever object does not expose a recognized retrieval method. "
        "Expected one of: invoke, get_relevant_documents, retrieve."
    )


class ConversationalRetrievalChain:
    """
    Custom ConversationalRetrievalChain compatible with LangChain 1.0+.
    """
    
    def __init__(self, llm, retriever, prompt_template: str, memory: Optional[Dict] = None):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.memory = memory or {"chat_history": []}
        
        # Create prompt
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )
        
        # Build chain using LCEL
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the retrieval chain using LCEL."""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def format_chat_history(history):
            if not history:
                return ""
            formatted = []
            for msg in history[-5:]:  # Last 5 messages
                if isinstance(msg, dict):
                    if "human" in msg:
                        formatted.append(f"Human: {msg['human']}")
                    if "ai" in msg:
                        formatted.append(f"AI: {msg['ai']}")
                elif isinstance(msg, tuple):
                    formatted.append(f"Human: {msg[0]}\nAI: {msg[1]}")
            return "\n".join(formatted)
        
        def retrieve_and_format(inputs: Dict[str, Any]) -> Dict[str, Any]:
            question = inputs["question"]
            chat_history = inputs.get("chat_history", [])
            
            # Retrieve documents (robust across LangChain versions)
            docs = _robust_retrieve(self.retriever, question)
            
            # Format context
            context = format_docs(docs)
            history_str = format_chat_history(chat_history)
            
            return {
                "context": context,
                "question": question,
                "chat_history": history_str,
                "source_documents": docs
            }
        
        # Create the chain
        # Handle both chat models (return messages) and LLM models (return strings)
        if hasattr(self.llm, 'invoke'):
            # Chat model - needs StrOutputParser
            chain = (
                RunnableLambda(lambda x: {"question": x["question"], "chat_history": x.get("chat_history", [])})
                | RunnableLambda(retrieve_and_format)
                | RunnableLambda(lambda x: self.prompt.format(**x))
                | self.llm
                | StrOutputParser()
            )
        else:
            # LLM model - returns string directly
            chain = (
                RunnableLambda(lambda x: {"question": x["question"], "chat_history": x.get("chat_history", [])})
                | RunnableLambda(retrieve_and_format)
                | RunnableLambda(lambda x: self.prompt.format(**x))
                | self.llm
            )
        
        return chain
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the chain."""
        question = inputs["question"]
        chat_history = inputs.get("chat_history", self.memory.get("chat_history", []))
        
        # Run chain
        answer = self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        # Get source documents from the retrieval step (robust)
        docs = _robust_retrieve(self.retriever, question)
        
        # Update memory
        self.memory["chat_history"].append((question, answer))
        if len(self.memory["chat_history"]) > 10:  # Keep last 10 messages
            self.memory["chat_history"] = self.memory["chat_history"][-10:]
        
        return {
            "answer": answer,
            "source_documents": docs
        }
    
    @classmethod
    def from_llm(cls, llm, retriever, memory=None, return_source_documents=True, verbose=False, combine_docs_chain_kwargs=None):
        """Create chain from LLM and retriever (compatible with old API)."""
        prompt_template = load_prompt_template()
        if combine_docs_chain_kwargs and "prompt" in combine_docs_chain_kwargs:
            prompt_template = combine_docs_chain_kwargs["prompt"].template
        
        return cls(llm=llm, retriever=retriever, prompt_template=prompt_template, memory=memory)


def create_chain(filters: Dict[str, Any] = None, k: int = None) -> ConversationalRetrievalChain:
    """
    Create ConversationalRetrievalChain with retriever and LLM.
    
    Args:
        filters: Metadata filters for retrieval
        k: Number of documents to retrieve
        
    Returns:
        ConversationalRetrievalChain instance
    """
    logger.info("Creating ConversationalRetrievalChain")
    
    # Get LLM
    llm = get_llm()
    
    # Get retriever
    retriever = get_retriever(filters=filters, k=k)
    
    # Load prompt template
    prompt_template = load_prompt_template()
    
    # Create memory
    memory = {"chat_history": []}
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])}
    )
    
    logger.info("Chain created successfully")
    return chain


def query_chain(
    chain: ConversationalRetrievalChain,
    query: str,
    chat_history: List[Tuple[str, str]] = None
) -> Dict[str, Any]:
    """
    Query the chain and return formatted response.
    
    Args:
        chain: ConversationalRetrievalChain instance
        query: User query
        chat_history: List of (question, answer) tuples
        
    Returns:
        Dictionary with answer, recommended cars, and sources
    """
    logger.info(f"Querying chain with: {query}")
    
    # Prepare input
    inputs = {"question": query}
    if chat_history:
        inputs["chat_history"] = chat_history
    
    # Run chain
    result = chain(inputs)
    
    # Extract answer and source documents
    answer = result.get("answer", "")
    source_documents = result.get("source_documents", [])
    
    # Extract recommended cars from source documents
    recommended = []
    sources = []
    
    for doc in source_documents:
        # Extract metadata from document
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        # Try to extract car information
        car_info = {}
        if "make" in metadata:
            car_info["make"] = metadata["make"]
        if "model" in metadata:
            car_info["model"] = metadata["model"]
        if "price" in metadata:
            car_info["price"] = metadata["price"]
        if "body_type" in metadata:
            car_info["body_type"] = metadata["body_type"]
        if "fuel_type" in metadata:
            car_info["fuel_type"] = metadata["fuel_type"]
        
        # Add all metadata as car info
        car_info.update(metadata)
        
        if car_info:
            recommended.append(car_info)
        
        # Add source info
        sources.append({
            "content": page_content[:200] + "..." if len(page_content) > 200 else page_content,
            "metadata": metadata
        })
    
    logger.info(f"Retrieved {len(recommended)} recommendations")
    
    return {
        "answer": answer,
        "recommended": recommended,
        "sources": sources
    }
