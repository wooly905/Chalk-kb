"""Responses API wrapper: structured outputs, response validation and image-input guards."""
from __future__ import annotations

import base64
import binascii
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

import jsonschema

from .config import MAX_OUTPUT_TOKENS_CAP, Settings

log = logging.getLogger("course_kb")


class LLMError(Exception):
    kind = "error"


class RefusalError(LLMError):
    kind = "refusal"


class IncompleteError(LLMError):
    kind = "incomplete"


class TruncatedError(IncompleteError):
    """Output hit max_output_tokens (reasoning tokens included); callers shrink the batch."""
    kind = "truncated"


class InputTooLargeError(LLMError):
    """HTTP 413: the estimated request exceeds a single deployment; callers shrink the input."""
    kind = "input_too_large"


# Errors that callers recover from by splitting the request into smaller pieces.
SPLITTABLE = (TruncatedError, InputTooLargeError)


class EmptyOutputError(LLMError):
    kind = "empty"


class InvalidJSONError(LLMError):
    kind = "invalid_json"


class SchemaValidationError(LLMError):
    kind = "schema"


class FatalLLMError(LLMError):
    """Authentication, unknown model or unsupported capability: never retried."""
    kind = "fatal"


class ImageInputError(ValueError):
    pass


_DATA_URL = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$")


def to_data_url(path: Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0]
    if mime is None or not mime.startswith("image/"):
        raise ImageInputError(f"Not an image file: {path}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def validate_image_part(part: dict) -> None:
    if "file_id" in part:
        raise ImageInputError("input_image must not use file_id")
    url = part.get("image_url")
    if not isinstance(url, str):
        raise ImageInputError("input_image.image_url must be a data:image/...;base64 URL")
    m = _DATA_URL.match(url)
    if not m:
        raise ImageInputError("input_image only accepts data:image/...;base64,... URLs "
                              "(http(s) URLs, file:// and local paths are rejected)")
    try:
        data = base64.b64decode(m.group(2), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageInputError(f"invalid base64 image data: {e}") from e
    if not data:
        raise ImageInputError("empty image data")


def validate_input(input_value: Any) -> None:
    """Reject any non-data-URL image before a request is sent."""
    if isinstance(input_value, str):
        return
    for msg in input_value:
        content = msg.get("content")
        if isinstance(content, str):
            continue
        for part in content or []:
            t = part.get("type")
            if t == "input_image":
                validate_image_part(part)
            elif t not in ("input_text", "output_text", "refusal"):
                raise ImageInputError(f"unsupported content type: {t}")


def build_text_format(name: str, schema: dict) -> dict:
    return {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}


def parse_response(response: Any, schema: dict) -> dict:
    """Validate a Responses API result and return the schema-checked JSON object."""
    status = getattr(response, "status", None)
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", None) or []:
                if getattr(c, "type", None) == "refusal":
                    raise RefusalError(getattr(c, "refusal", "refused"))
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) if details is not None else None
        if reason == "max_output_tokens":
            raise TruncatedError("response truncated at max_output_tokens")
        raise IncompleteError(f"response not completed: status={status} reason={reason}")
    text = getattr(response, "output_text", None) or ""
    if not text.strip():
        raise EmptyOutputError("empty output_text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidJSONError(str(e)) from e
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        raise SchemaValidationError(e.message) from e
    return data


class OpenAILLM:
    """Real client: client.responses.create(); keys come only from OPENAI_API_KEY."""

    def __init__(self, settings: Settings):
        import os
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise FatalLLMError("OPENAI_API_KEY is not set")
        self.settings = settings
        self.client = OpenAI(api_key=key, base_url=settings.base_url)
        if not hasattr(self.client, "responses"):
            raise FatalLLMError("openai SDK >= 1.66 is required (client.responses missing)")

    def structured(self, role: str, name: str, schema: dict, instructions: str, input: Any,
                   max_output_tokens: int = MAX_OUTPUT_TOKENS_CAP) -> tuple[dict, dict]:
        import openai

        model = self.settings.model_for(role)
        if not model:
            raise FatalLLMError("COURSE_KB_MODEL is not set")
        validate_input(input)
        try:
            resp = self.client.responses.create(
                model=model,
                instructions=instructions,
                input=input,
                text=build_text_format(name, schema),
                max_output_tokens=min(max_output_tokens, MAX_OUTPUT_TOKENS_CAP),
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise FatalLLMError(f"{type(e).__name__}: {e}") from e
        except openai.BadRequestError as e:
            raise FatalLLMError(f"bad request / unsupported capability: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code == 413:
                raise InputTooLargeError(f"request too large: {e}") from e
            raise LLMError(f"{type(e).__name__}: {e}") from e
        except openai.APIError as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        data = parse_response(resp, schema)
        return data, {"requested_model": model, "actual_model": getattr(resp, "model", model)}


def describe_error(e: BaseException) -> str:
    kind = getattr(e, "kind", type(e).__name__)
    return f"{kind}: {e}"
