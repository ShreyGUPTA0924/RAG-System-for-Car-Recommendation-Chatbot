"""
LLM wrapper for Ollama (local) or hosted Llama API.
Provides LangChain-compatible interface.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
try:
    from langchain_community.llms import Ollama
    from langchain_core.language_models.llms import BaseLLM as LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.llms import Ollama
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
    except ImportError:
        from langchain_community.llms import Ollama
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
HF_API_KEY = os.getenv("HF_API_KEY", None)
HF_MODEL_ENDPOINT = os.getenv("HF_MODEL_ENDPOINT", None)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))


class HostedLlamaLLM(LLM):
    """
    LangChain-compatible wrapper for hosted Llama API (e.g., HuggingFace Inference API).
    """
    
    endpoint: str = Field(default="")
    api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    
    @property
    def _llm_type(self) -> str:
        return "hosted_llama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the hosted LLM API."""
        return self._generate([prompt], stop=stop, run_manager=run_manager, **kwargs).generations[0][0].text
    
    def _generate(
        self,
        prompts: list,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ):
        """Generate responses for prompts (required by LangChain LLM base class)."""
        from langchain_core.outputs import Generation, LLMResult
        
        generations = []
        for prompt in prompts:
            try:
                headers = {
                    "Content-Type": "application/json"
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                # Try different payload formats for compatibility
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_new_tokens": 256,
                        "return_full_text": False
                    }
                }
                
                response = httpx.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=60.0  # Longer timeout for model loading
                )
                
                # Handle 503 (model loading) - wait and retry once
                if response.status_code == 503:
                    logger.info("Model is loading, waiting 10 seconds...")
                    import time
                    time.sleep(10)
                    response = httpx.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                        timeout=60.0
                    )
                
                # Handle 410 (Gone) - model no longer available
                if response.status_code == 410:
                    error_msg = (
                        f"Model endpoint returned 410 Gone. The model at {self.endpoint} is no longer available.\n"
                        f"Please install Ollama (recommended) or update HF_MODEL_ENDPOINT in .env to a valid model."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                response.raise_for_status()
                result = response.json()
                
                # Handle different response formats
                text = ""
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        text = result[0]["generated_text"]
                    elif isinstance(result[0], dict) and "text" in result[0]:
                        text = result[0]["text"]
                    elif isinstance(result[0], str):
                        text = result[0]
                    elif "summary_text" in result[0]:
                        text = result[0]["summary_text"]
                
                # Fallback: try to extract text from response
                if not text and isinstance(result, dict):
                    if "generated_text" in result:
                        text = result["generated_text"]
                    elif "text" in result:
                        text = result["text"]
                    elif "summary_text" in result:
                        text = result["summary_text"]
                
                if not text:
                    logger.warning(f"Unexpected response format: {result}")
                    text = str(result)
                
                generations.append([Generation(text=text)])
                
            except ValueError as ve:
                # Re-raise ValueError (contains 410 Gone error with helpful message)
                raise
            except Exception as e:
                logger.error(f"Error calling hosted LLM: {e}")
                # Return error message as generation for other errors
                generations.append([Generation(text=f"Error: {str(e)}")])
        
        return LLMResult(generations=generations)


def get_llm() -> LLM:
    """
    Get LLM instance based on environment configuration.
    Prioritizes hosted API if configured, falls back to Ollama.
    
    Returns:
        LangChain LLM instance
    """
    # Prioritize Ollama if available (most reliable for local development)
    if OLLAMA_URL:
        try:
            logger.info(f"Initializing Ollama LLM at {OLLAMA_URL} with model {OLLAMA_MODEL}")
            # Check if Ollama is reachable first
            import httpx
            try:
                response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
                response.raise_for_status()
                logger.info("Ollama is reachable")
            except Exception as conn_error:
                logger.warning(f"Ollama not reachable at {OLLAMA_URL}: {conn_error}")
                logger.info("To use Ollama: 1) Install from https://ollama.ai 2) Run 'ollama serve' 3) Run 'ollama pull llama2'")
                raise ValueError(f"Ollama not available at {OLLAMA_URL}. Please start Ollama or use HuggingFace API.")
            
            from langchain_community.llms import Ollama
            llm = Ollama(
                base_url=OLLAMA_URL,
                model=OLLAMA_MODEL,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P
            )
            # Test connection
            try:
                test_result = llm.invoke("test", config={"callbacks": []})
                logger.info("Ollama LLM initialized and tested successfully")
                return llm
            except Exception as test_error:
                logger.warning(f"Ollama connection test failed: {test_error}")
                logger.info(f"Make sure model '{OLLAMA_MODEL}' is installed: ollama pull {OLLAMA_MODEL}")
                raise
        except ValueError:
            # Re-raise ValueError (contains helpful instructions)
            if not (HF_API_KEY and HF_MODEL_ENDPOINT):
                raise
            logger.info("Ollama not available, trying HuggingFace API...")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama: {e}")
            if not (HF_API_KEY and HF_MODEL_ENDPOINT):
                raise ValueError(
                    "No LLM available. Please configure either:\n"
                    "1. Ollama: Install from https://ollama.ai, run 'ollama serve', then 'ollama pull llama2'\n"
                    "2. HuggingFace: Set HF_API_KEY and HF_MODEL_ENDPOINT in .env"
                )
    
    # Fall back to hosted HuggingFace API if configured
    if HF_API_KEY and HF_MODEL_ENDPOINT:
        logger.info(f"Initializing hosted LLM at {HF_MODEL_ENDPOINT}")
        try:
            # Extract model name if full URL provided
            model_name = HF_MODEL_ENDPOINT
            if "/models/" in HF_MODEL_ENDPOINT:
                model_name = HF_MODEL_ENDPOINT.split("/models/")[-1]
                endpoint = HF_MODEL_ENDPOINT
            else:
                # Assume it's just a model name, construct endpoint
                endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
            
            llm = HostedLlamaLLM(
                endpoint=endpoint,
                api_key=HF_API_KEY,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P
            )
            # Test with a simple call to catch 410 errors early
            try:
                test_result = llm._generate(["test"], stop=None)
                logger.info("Hosted LLM initialized successfully")
                return llm
            except ValueError as ve:
                # Re-raise ValueError (contains helpful message about 410 Gone)
                raise
            except Exception as test_error:
                error_str = str(test_error)
                if "410" in error_str or "Gone" in error_str:
                    raise ValueError(
                        f"❌ HuggingFace model '{model_name}' is no longer available (410 Gone).\n\n"
                        f"✅ SOLUTION: Install Ollama (recommended for local development):\n"
                        f"   1. Download from https://ollama.ai\n"
                        f"   2. Run: ollama serve\n"
                        f"   3. Run: ollama pull llama2\n"
                        f"   4. The system will automatically use Ollama\n\n"
                        f"OR update HF_MODEL_ENDPOINT in .env to a different model:\n"
                        f"   Visit https://huggingface.co/models?pipeline_tag=text-generation\n"
                        f"   Find an available model and update the endpoint"
                    )
                raise
        except ValueError:
            # Re-raise ValueError (contains helpful message)
            raise
        except Exception as e:
            logger.error(f"Failed to initialize hosted LLM: {e}")
            raise ValueError(
                "No LLM available. Please configure either:\n"
                "1. Ollama (recommended): Install from https://ollama.ai, run 'ollama serve', then 'ollama pull llama2'\n"
                "2. HuggingFace: Update HF_MODEL_ENDPOINT in .env to a valid model endpoint"
            )
    
    # No LLM configured
    if OLLAMA_URL:
        try:
            logger.info(f"Initializing Ollama LLM at {OLLAMA_URL} with model {OLLAMA_MODEL}")
            # Check if Ollama is reachable first
            import httpx
            try:
                response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
                response.raise_for_status()
            except Exception:
                raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_URL}. Is Ollama running?")
            
            try:
                # Try ChatOllama first (LangChain 1.0+)
                from langchain_community.chat_models import ChatOllama
                from langchain_core.language_models.chat_models import BaseChatModel
                llm = ChatOllama(
                    base_url=OLLAMA_URL,
                    model=OLLAMA_MODEL,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P
                )
                logger.info("Ollama connection successful (ChatOllama)")
                return llm
            except ImportError:
                # Fallback to Ollama LLM (older versions)
                llm = Ollama(
                    base_url=OLLAMA_URL,
                    model=OLLAMA_MODEL,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P
                )
                logger.info("Ollama connection successful (Ollama LLM)")
                return llm
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            if not (HF_API_KEY and HF_MODEL_ENDPOINT):
                raise
    
    # If neither is configured, raise error
    raise ValueError(
        "No LLM configured. Please set either:\n"
        "- OLLAMA_URL and OLLAMA_MODEL for local Ollama, or\n"
        "- HF_API_KEY and HF_MODEL_ENDPOINT for hosted API\n\n"
        "Note: For testing without LLM, you can use a mock LLM in tests."
    )

