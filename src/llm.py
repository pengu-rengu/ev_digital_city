import json
from typing import TypeVar
import ollama
from pydantic import BaseModel, ValidationError

MODEL = "gemma4:e4b"

StructuredT = TypeVar("StructuredT", bound = BaseModel)

def call_llm(client: ollama.Client, messages: list[dict], schema: type[StructuredT] | None = None) -> StructuredT | str:
    input_messages = messages
    format_schema = None
    if schema is not None:
        schema_text = f"\n\nRespond with only a single JSON object matching this JSON schema. No markdown fences, no extra text:\n{json.dumps(schema.model_json_schema())}"
        input_messages = [{**messages[0], "content": messages[0]["content"] + schema_text}] + messages[1:]
        format_schema = schema.model_json_schema()
    failures = 0
    while True:
        try:
            response = client.chat(
                model = MODEL,
                messages = input_messages,
                format = format_schema
            )
            text = response.message.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            if schema is None:
                return text
            return schema.model_validate_json(text)
        except ValidationError:
            failures += 1
            print(f"Failed to parse:\n{text}\n\nRetrying...")
            continue
        except Exception as error:
            failures += 1
            print(f"LLM call failed:\n{error}\n\nRetrying...")
            continue
