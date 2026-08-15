# Prompt Batch parsing core — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0, like the rest of the pack.
#
# Pure (comfy-free) parse/select logic shared by the Prompt Batch node (nodes/prompt_batch.py) and
# the Civitai metadata tracer (nodes/civitai_metadata/tracer.py). Mirrored in web/js/prompt_batch.js
# — keep the three in sync.
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS_JSON = json.dumps(
    [
        {"positive": "a red fox in the snow, highly detailed, 8k", "negative": "blurry, watermark"},
        {"positive": "a neon-lit city street at night, cinematic", "negative": ""},
    ],
    indent=2,
)


def parse_prompts(raw):
    """Parse the prompts JSON into a list of ``(positive, negative)`` string pairs.

    Accepts a JSON array of objects (canonical), a bare string entry (positive-only shorthand), or a
    ``{"prompts": [...]}`` wrapper. Raises ``ValueError`` with a human-readable message on anything
    malformed.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Prompt JSON is empty — paste a JSON array of prompts.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e.msg} (line {e.lineno}, column {e.colno}).") from e

    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        data = data["prompts"]
    if not isinstance(data, list):
        raise ValueError('Expected a JSON array of prompts, e.g. [{"positive": "..."}, ...].')
    if not data:
        raise ValueError("Prompt list is empty — it needs at least one prompt.")

    prompts = []
    for i, entry in enumerate(data):
        where = f"Prompt #{i + 1}"
        if isinstance(entry, str):
            positive, negative = entry, ""
        elif isinstance(entry, dict):
            positive = entry.get("positive", entry.get("prompt", ""))
            negative = entry.get("negative", "")
            if negative is None:
                negative = ""
            if not isinstance(positive, str):
                raise ValueError(f"{where}: 'positive' must be a string.")
            if not isinstance(negative, str):
                raise ValueError(f"{where}: 'negative' must be a string.")
        else:
            raise ValueError(f"{where}: each entry must be an object with a 'positive' field (or a plain string).")
        if not positive.strip():
            raise ValueError(f"{where}: 'positive' is empty.")
        prompts.append((positive, negative))
    return prompts


def select_prompt(prompts, index):
    """Pick the prompt at ``index``, clamping into range (never raises).

    Returns ``(positive, negative, used_index)``.
    """
    count = len(prompts)
    try:
        idx = int(index)
    except (TypeError, ValueError):
        logger.warning("ITL Prompt Batch: index %r is not an int, using 0.", index)
        idx = 0
    if idx < 0:
        logger.warning("ITL Prompt Batch: index %d < 0, clamping to 0.", idx)
        idx = 0
    elif idx >= count:
        logger.warning("ITL Prompt Batch: index %d >= count %d, clamping to last prompt.", idx, count)
        idx = count - 1
    positive, negative = prompts[idx]
    return positive, negative, idx
