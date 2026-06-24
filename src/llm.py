import json
import time
from typing import TypeVar
from openai import OpenAI
from pydantic import BaseModel, ValidationError

MODEL = "openai/gpt-oss-120b:free"

StructuredT = TypeVar("StructuredT", bound = BaseModel)

def call_llm(client: OpenAI, messages: list[dict], schema: type[StructuredT] | None = None, reasoning: dict | None = None) -> StructuredT | str:
    input_messages = messages
    if schema is not None:
        schema_text = f"\n\nRespond with only a single JSON object matching this JSON schema. No markdown fences, no extra text:\n{json.dumps(schema.model_json_schema())}"
        input_messages = [{**messages[0], "content": messages[0]["content"] + schema_text}] + messages[1:]
    while True:
        time.sleep(3)
        response = client.responses.create(
            model = MODEL,
            input = input_messages,
            **({"reasoning": reasoning} if reasoning is not None else {})
        )
        text = response.output_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if schema is None:
            return text
        try:
            return schema.model_validate_json(text)
        except ValidationError:
            print(f"Failed to parse:\n{text}\n\nRetrying...")
            continue
