import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm import MODEL, call_llm


class Answer(BaseModel):
    choice: str


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    def chat(self, model: str, messages: list[dict], format: dict | None = None) -> FakeResponse:
        self.calls.append({"model": model, "messages": messages, "format": format})
        return FakeResponse(self.outputs[len(self.calls) - 1])


def test_call_llm_returns_plain_text() -> None:
    client = FakeClient(["  hello  "])

    result = call_llm(client, [{"role": "user", "content": "Say hello."}])

    assert result == "hello"
    assert client.calls[0]["model"] == MODEL
    assert client.calls[0]["format"] is None


def test_call_llm_passes_schema_format() -> None:
    client = FakeClient(['{"choice":"A"}'])

    result = call_llm(client, [{"role": "system", "content": "Pick."}], schema = Answer)

    assert result == Answer(choice = "A")
    assert client.calls[0]["format"] == Answer.model_json_schema()
    assert "Respond with only a single JSON object" in client.calls[0]["messages"][0]["content"]


def test_call_llm_retries_invalid_json_immediately() -> None:
    client = FakeClient(["not json", '{"choice":"B"}'])

    result = call_llm(client, [{"role": "system", "content": "Pick."}], schema = Answer)

    assert result == Answer(choice = "B")
    assert len(client.calls) == 2
