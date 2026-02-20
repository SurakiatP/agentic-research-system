"""
DeepSeek Judge Model for DeepEval
=================================
Wraps deepseek-reasoner (V3.2 Thinking Mode) as an LLM-as-a-Judge
for evaluation metrics like Faithfulness, Relevancy, etc.

The Thinking Mode provides stronger reasoning than standard deepseek-chat,
making it a better evaluator while using the same API key.
"""

import json
from typing import Optional, Union, Any
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI

from config.settings import settings
from src.utils.logger import logger


class DeepSeekJudge:
    """
    DeepSeek Reasoner (V3.2 Thinking Mode) as LLM Judge for DeepEval.
    
    Inherits from DeepEvalBaseLLM to integrate seamlessly with all
    DeepEval metrics (Faithfulness, Relevancy, ToolCorrectness, etc.).
    """

    def __init__(self, model_name: Optional[str] = None):
        # Load from model_config.yaml
        judge_cfg = settings.model_config_yaml.get("judge_model", {})
        self._model_name = model_name or judge_cfg.get("name", "deepseek-reasoner")
        api_base = judge_cfg.get("api_base", "https://api.deepseek.com")
        
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=api_base,
        )
        self.async_client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=api_base,
        )
        logger.info(f"DeepSeekJudge initialized with model: {self._model_name}")

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self):
        return self._model_name

    def generate(self, prompt: str, schema: Optional[Any] = None) -> Union[str, BaseModel]:
        """
        Generate a response from the judge model.
        If schema is provided, attempts to return structured output.
        
        Note: deepseek-reasoner returns reasoning_content (thinking) 
        separately from content (final answer), so content is always clean.
        """
        messages = [{"role": "user", "content": prompt}]
        
        # If schema is provided, instruct the model to output JSON
        if schema and hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
            messages[0]["content"] += (
                "\n\nYou MUST respond with ONLY valid JSON (no markdown, no extra text) "
                f"matching this schema:\n{json.dumps(json_schema, indent=2)}"
            )

        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            
            # If schema is provided, parse the JSON response
            if schema and hasattr(schema, "model_validate"):
                # Strip markdown code fences if present
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    cleaned = cleaned.rsplit("```", 1)[0]
                data = json.loads(cleaned)
                return schema.model_validate(data)
            
            return content
            
        except Exception as e:
            logger.error(f"DeepSeekJudge.generate failed: {e}")
            raise

    async def a_generate(self, prompt: str, schema: Optional[Any] = None) -> Union[str, BaseModel]:
        """Async version of generate."""
        messages = [{"role": "user", "content": prompt}]
        
        if schema and hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
            messages[0]["content"] += (
                "\n\nYou MUST respond with ONLY valid JSON (no markdown, no extra text) "
                f"matching this schema:\n{json.dumps(json_schema, indent=2)}"
            )

        try:
            response = await self.async_client.chat.completions.create(
                model=self._model_name,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            
            if schema and hasattr(schema, "model_validate"):
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    cleaned = cleaned.rsplit("```", 1)[0]
                data = json.loads(cleaned)
                return schema.model_validate(data)
            
            return content
            
        except Exception as e:
            logger.error(f"DeepSeekJudge.a_generate failed: {e}")
            raise
