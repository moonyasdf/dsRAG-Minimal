# dsrag_minimal/core/llm.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Iterator, Union
import os
import ollama
from openai import OpenAI, Stream
from anthropic import Anthropic, Stream as AnthropicStream
import instructor
from pydantic import BaseModel
import warnings

# Listas de modelos simplificadas (podrían cargarse desde utils si fuera necesario)
OPENAI_MODEL_NAMES = ["gpt-4o-mini", "gpt-4o"] # Ejemplo mínimo
ANTHROPIC_MODEL_NAMES = ["claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"] # Ejemplo mínimo
OLLAMA_MODEL_NAMES = ["llama3", "mistral"] # Ejemplo mínimo

class LLM(ABC):
    """Clase base abstracta para modelos de lenguaje."""
    @abstractmethod
    def generate(self, messages: List[Dict], stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """Genera una respuesta (string o stream) a partir de mensajes."""
        pass

    @abstractmethod
    def generate_structured(self, messages: List[Dict], response_model: BaseModel, stream: bool = False, **kwargs) -> Union[BaseModel, Iterator[BaseModel]]:
        """Genera una respuesta estructurada (objeto Pydantic o stream)"""
        pass

    def to_dict(self):
        """Serializa la configuración del LLM."""
        return {"subclass_name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict) -> 'LLM':
        """Crea una instancia de LLM desde la configuración."""
        subclass_name = config.get("subclass_name")
        # Mapeo explícito para las clases incluidas
        if subclass_name == "OllamaLLM":
            return OllamaLLM(**config)
        elif subclass_name == "OpenAILLM":
            return OpenAILLM(**config)
        elif subclass_name == "AnthropicLLM":
            return AnthropicLLM(**config)
        else:
            raise ValueError(f"Unknown or unsupported LLM subclass: {subclass_name}")

# --- Implementaciones Específicas ---

class OllamaLLM(LLM):
    """Implementación de LLM usando Ollama."""
    def __init__(self, model: str = "llama3", temperature: float = 0.0, client: Optional[ollama.Client] = None, **kwargs):
        self.model = model
        self.temperature = temperature
        self.client = client or ollama.Client()
        # Verifica/descarga el modelo
        try:
            print(f"Checking/Pulling Ollama model: {model}...")
            ollama.pull(model)
            print(f"Model {model} available.")
        except Exception as e:
            print(f"Warning: Could not pull Ollama model '{model}'. Ensure it's available. Error: {e}")

    def generate(self, messages: List[Dict], stream: bool = False, max_tokens: int = 4000, **kwargs) -> Union[str, Iterator[str]]:
        """Genera respuesta de texto con Ollama."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=stream,
                options={"temperature": self.temperature, "num_predict": max_tokens}
            )
            if stream:
                return self._process_stream(response)
            else:
                return response['message']['content']
        except Exception as e:
            print(f"Error generating with Ollama model {self.model}: {e}")
            return "" if not stream else iter([]) # Devuelve vacío o iterador vacío en error

    def _process_stream(self, stream_response: Iterator[Dict]) -> Iterator[str]:
        """Procesa el stream de Ollama."""
        for chunk in stream_response:
            if chunk.get('message', {}).get('content'):
                yield chunk['message']['content']

    def generate_structured(self, messages: List[Dict], response_model: BaseModel, stream: bool = False, max_tokens: int = 4000, **kwargs) -> Union[BaseModel, Iterator[BaseModel]]:
        """Genera respuesta estructurada con Ollama usando Instructor."""
        # Instructor no tiene soporte directo para streaming con Ollama AFAIK.
        # Se implementa sin streaming por ahora, o se puede intentar parsear JSON en el stream.
        # Implementación simplificada sin streaming para Ollama con Instructor:
        if stream:
            warnings.warn("Streaming structured output is not directly supported for Ollama via Instructor in this minimal version. Returning non-streamed.")

        # Intenta usar el modo JSON si el modelo lo soporta
        try:
            client_instructor = instructor.from_ollama(self.client, mode=instructor.Mode.JSON) # o instructor.Mode.TOOLS
            response = client_instructor.chat.completions.create(
                model=self.model,
                response_model=response_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            print(f"Error generating structured output with Ollama {self.model}: {e}. Trying standard generation and parsing.")
            # Fallback: Generación estándar y parseo manual (menos fiable)
            text_response = self.generate(messages, stream=False, max_tokens=max_tokens)
            try:
                # Intenta parsear la respuesta como JSON
                import json
                parsed_data = json.loads(text_response)
                return response_model.model_validate(parsed_data)
            except Exception as parse_error:
                print(f"Failed to parse Ollama response into {response_model.__name__}: {parse_error}")
                # Devuelve una instancia vacía o maneja el error como sea apropiado
                try:
                    return response_model() # Intenta crear instancia por defecto
                except:
                     raise RuntimeError(f"Could not generate or parse structured response from Ollama: {e}") from parse_error


    def to_dict(self):
        """Serializa configuración de Ollama."""
        return {"subclass_name": "OllamaLLM", "model": self.model, "temperature": self.temperature}

# --- Clases para APIs (OpenAI, Anthropic) - Simplificadas ---
# Se mantienen como ejemplos, pero la inicialización requiere inyección explícita

class OpenAILLM(LLM):
    """Implementación LLM usando API OpenAI (ejemplo)."""
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("DSRAG_OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API key needed.")
        client_args = {"api_key": self.api_key}
        if self.base_url: client_args["base_url"] = self.base_url
        self.client = OpenAI(**client_args)
        self.instructor_client = instructor.from_openai(self.client)

    def generate(self, messages: List[Dict], stream: bool = False, max_tokens: int = 4000, **kwargs) -> Union[str, Iterator[str]]:
        """Genera texto con OpenAI."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            if stream:
                return self._process_openai_stream(completion)
            else:
                return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating with OpenAI model {self.model}: {e}")
            return "" if not stream else iter([])

    def _process_openai_stream(self, stream: Stream) -> Iterator[str]:
        """Procesa el stream de OpenAI."""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_structured(self, messages: List[Dict], response_model: BaseModel, stream: bool = False, max_tokens: int = 4000, **kwargs) -> Union[BaseModel, Iterator[BaseModel]]:
        """Genera respuesta estructurada con OpenAI."""
        try:
            # Determina el modelo parcial para streaming
            partial_model = instructor.Partial[response_model] if stream else response_model

            completion = self.instructor_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=partial_model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            return completion # Devuelve objeto o stream
        except Exception as e:
            print(f"Error generating structured output with OpenAI {self.model}: {e}")
            if stream: return iter([])
            try: return response_model() # Intenta instancia por defecto
            except: raise RuntimeError(f"Could not generate/parse structured response from OpenAI: {e}") from e

    def to_dict(self):
        return {"subclass_name": "OpenAILLM", "model": self.model, "temperature": self.temperature, "base_url": self.base_url}


class AnthropicLLM(LLM):
    """Implementación LLM usando API Anthropic (ejemplo)."""
    def __init__(self, model: str = "claude-3-haiku-20240307", temperature: float = 0.0, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.environ.get("DSRAG_ANTHROPIC_BASE_URL")
        if not self.api_key:
            raise ValueError("Anthropic API key needed.")
        client_args = {"api_key": self.api_key}
        if self.base_url: client_args["base_url"] = self.base_url
        self.client = Anthropic(**client_args)
        self.instructor_client = instructor.from_anthropic(self.client, mode=instructor.Mode.ANTHROPIC_JSON)

    # Helper para extraer system prompt
    def _extract_system_prompt(self, messages: List[Dict]) -> (Optional[str], List[Dict]):
        system = None
        user_assistant_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"] # Asume string simple por ahora
            else:
                user_assistant_messages.append(m)
        return system, user_assistant_messages

    def generate(self, messages: List[Dict], stream: bool = False, max_tokens: int = 4000, **kwargs) -> Union[str, Iterator[str]]:
        """Genera texto con Anthropic."""
        system, user_assistant_messages = self._extract_system_prompt(messages)
        api_kwargs = {
            "model": self.model,
            "messages": user_assistant_messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if system: api_kwargs["system"] = system

        try:
            response = self.client.messages.create(**api_kwargs)
            if stream:
                return self._process_anthropic_stream(response)
            else:
                # Asume que el contenido es texto
                return "".join(block.text for block in response.content if hasattr(block, 'text'))
        except Exception as e:
            print(f"Error generating with Anthropic model {self.model}: {e}")
            return "" if not stream else iter([])

    def _process_anthropic_stream(self, stream: AnthropicStream) -> Iterator[str]:
        """Procesa el stream de Anthropic."""
        for event in stream:
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                yield event.delta.text

    def generate_structured(self, messages: List[Dict], response_model: BaseModel, stream: bool = False, max_tokens: int = 4000, **kwargs) -> Union[BaseModel, Iterator[BaseModel]]:
        """Genera respuesta estructurada con Anthropic."""
        system, user_assistant_messages = self._extract_system_prompt(messages)
        partial_model = instructor.Partial[response_model] if stream else response_model

        api_kwargs = {
            "model": self.model,
            "messages": user_assistant_messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "response_model": partial_model,
            "stream": stream,
        }
        if system: api_kwargs["system"] = system

        try:
            response = self.instructor_client.messages.create(**api_kwargs)
            return response # Devuelve objeto o stream
        except Exception as e:
            print(f"Error generating structured output with Anthropic {self.model}: {e}")
            if stream: return iter([])
            try: return response_model() # Instancia por defecto
            except: raise RuntimeError(f"Could not generate/parse structured response from Anthropic: {e}") from e

    def to_dict(self):
        return {"subclass_name": "AnthropicLLM", "model": self.model, "temperature": self.temperature, "base_url": self.base_url}


# Función get_response refactorizada para usar inyección de instancia LLM
def get_response_via_instance(
    llm_instance: LLM, # Instancia de LLM (OllamaLLM, OpenAILLM, etc.)
    messages: Optional[List[Dict]] = None,
    prompt: Optional[str] = None,
    response_model: Optional[BaseModel] = None,
    stream: bool = False,
    **kwargs # Pasa kwargs adicionales (como temperature, max_tokens)
) -> Any:
    """
    Genera una respuesta usando una instancia LLM proporcionada.
    Simplificado para eliminar la lógica de detección de proveedor interna.
    """
    if not messages and not prompt:
        raise ValueError("Either messages or prompt must be provided")
    if messages and prompt:
        warnings.warn("Both messages and prompt provided - using messages")

    final_messages = messages or [{"role": "user", "content": prompt}]

    # Pasa los kwargs directamente a la instancia
    if response_model:
        return llm_instance.generate_structured(final_messages, response_model, stream=stream, **kwargs)
    else:
        return llm_instance.generate(final_messages, stream=stream, **kwargs)
