from groq import Groq
import openai

def initialize_groq_client(api_key):
    """Initialize and return the Groq client."""
    if not api_key:
        raise ValueError("API key is required to initialize Groq client.")
    return Groq(api_key=api_key)

def initialize_openai_client(api_key):
    """Initialize and return the OpenAI client."""
    if not api_key:
        raise ValueError("API key is required to initialize OpenAI client.")
    openai.api_key = api_key
    return openai
