import json
import boto3
import botocore
import config
from typing import Optional

# Initialize Bedrock client once
bedrock = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)

def _extract_text(result: dict) -> str:
    """
    Handles both common Bedrock Claude response shapes:
    1) invoke_model (Claude Messages): {"content": [{"type":"text","text":"..."}], ...}
    2) converse-style shape: {"output":{"message":{"content":[{"type":"text","text":"..."}]}}}
    """
    # Shape 1: Claude Messages via invoke_model
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        parts = [c.get("text", "") for c in result["content"] if c.get("type") == "text"]
        text = " ".join(p.strip() for p in parts if p.strip())
        if text:
            return text

    # Shape 2: Converse-like
    output = result.get("output", {})
    if isinstance(output, dict):
        msg = output.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            text = " ".join(p.strip() for p in parts if p.strip())
            if text:
                return text

    return ""

def query_bedrock(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """
    Calls Anthropic Claude on Bedrock via invoke_model using the Messages API schema.
    """
    if model_id is None:
        model_id = config.BEDROCK_MODEL_ID  # e.g. "anthropic.claude-3-5-sonnet-20240620-v1:0"

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    try:
        resp = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )

        # resp["body"] is a stream-like object
        raw = resp["body"].read() if hasattr(resp.get("body"), "read") else resp.get("body")
        result = json.loads(raw)

        text = _extract_text(result)
        return text or "⚠ No valid text content returned by the model."

    except botocore.exceptions.ClientError as e:
        # AWS-side error (auth, permissions, wrong model ID, etc.)
        return f"Error (AWS ClientError): {e.response.get('Error', {}).get('Message', str(e))}"
    except Exception as e:
        return f"Error: {str(e)}"
