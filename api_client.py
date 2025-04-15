"""
API Client wrapper for OpenAI interactions with comprehensive logging.

This module provides a wrapper around the OpenAI API client specifically for gpt-4-turbo
with proper logging of inputs and outputs. It serves as a standardized interface for
making API calls to OpenAI's language models within the ToP (Template-oriented Prompting) 
framework.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
import os
from datetime import datetime

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openai_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("openai_api")

class ApiClient:
    """
    A wrapper around the OpenAI API client for gpt-4-turbo with proper logging.
    
    This class provides methods to interact with OpenAI's API, specifically
    focusing on the gpt-4-turbo model, while ensuring comprehensive logging
    of all inputs and outputs for debugging and monitoring purposes.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: str = "https://api.openai.com/v1/",
        model: str = "gpt-4-turbo", 
        log_dir: str = "logs"
    ):
        """
        Initialize the ApiClient.
        
        Args:
            api_key (Optional[str]): The OpenAI API key. If None, will try to use 
                                     the OPENAI_API_KEY environment variable.
            base_url (str): The base URL for the OpenAI API.
            model (str): The model to use. Defaults to "gpt-4-turbo".
            log_dir (str): Directory to save detailed logs to.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via OPENAI_API_KEY environment variable")
        
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Ensure log directory exists
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Initialize conversation log
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"conversation_{self.conversation_id}.jsonl")
        
        logger.info(f"Initialized ApiClient with model {self.model}")

    def _log_conversation(self, messages: List[Dict[str, str]], response: ChatCompletionMessage, metadata: Dict[str, Any]) -> None:
        """
        Log the conversation to a JSONL file.
        
        Args:
            messages (List[Dict[str, str]]): The input messages.
            response (ChatCompletionMessage): The response from the API.
            metadata (Dict[str, Any]): Additional metadata for the request.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "messages": messages,
            "response": response.content,
            "metadata": metadata
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        logger.debug(f"Logged conversation to {self.log_file}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        streaming: bool = False,
        seed: Optional[int] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> ChatCompletionMessage:
        """
        Send a chat completion request to the OpenAI API.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the API.
            temperature (float): Controls randomness. Lower values make responses more deterministic.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            top_p (float): Controls diversity via nucleus sampling.
            frequency_penalty (float): Decreases likelihood of repeating tokens.
            presence_penalty (float): Increases likelihood of introducing new tokens.
            stop (Optional[Union[str, List[str]]]): Sequences where the API will stop generating.
            n (int): Number of completions to generate.
            streaming (bool): Whether to stream the response.
            seed (Optional[int]): A seed for deterministic sampling.
            retry_count (int): Number of retries on failure.
            retry_delay (float): Delay between retries in seconds.
            
        Returns:
            ChatCompletionMessage: The response message from the API.
            
        Raises:
            Exception: If the API request fails after all retries.
        """
        logger.info(f"Sending request to {self.model}")
        logger.debug(f"Input messages: {json.dumps(messages)}")
        
        metadata = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "n": n,
            "streaming": streaming,
            "seed": seed
        }
        
        attempt = 0
        while attempt < retry_count:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    n=n,
                    stream=streaming,
                    seed=seed
                )
                end_time = time.time()
                
                if streaming:
                    # Handle streaming response
                    # This is a placeholder - actual implementation would need to handle streaming chunks
                    logger.warning("Streaming mode is not fully implemented in logging")
                    return response
                
                # Log full response for debug purposes
                logger.debug(f"Full response: {response}")
                
                # Log the timing information
                duration = end_time - start_time
                logger.info(f"Request completed in {duration:.2f} seconds")
                
                # Log a more concise version of the input/output for normal logging
                logger.info(f"Input: {messages[-1]['content'][:100]}...")
                logger.info(f"Output: {response.choices[0].message.content[:100]}...")
                
                # Log the full conversation to file
                self._log_conversation(messages, response.choices[0].message, metadata)
                
                return response.choices[0].message
                
            except Exception as e:
                attempt += 1
                logger.warning(f"API request failed (attempt {attempt}/{retry_count}): {str(e)}")
                if attempt < retry_count:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed after {retry_count} attempts: {str(e)}")
                    raise

    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response to a prompt using the chat completion API.
        
        This is a simplified interface for single-prompt generation.
        
        Args:
            prompt (str): The user prompt to send to the API.
            system_prompt (Optional[str]): Optional system instructions.
            **kwargs: Additional arguments to pass to the chat_completion method.
            
        Returns:
            str: The generated text response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(messages, **kwargs)
        return response.content

    def generate_with_history(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Generate a response based on a conversation history.
        
        Args:
            messages (List[Dict[str, str]]): The conversation history.
            **kwargs: Additional arguments to pass to the chat_completion method.
            
        Returns:
            str: The generated text response.
        """
        response = self.chat_completion(messages, **kwargs)
        return response.content