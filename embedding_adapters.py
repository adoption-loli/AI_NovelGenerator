# embedding_adapters.py
# -*- coding: utf-8 -*-
import logging
import requests
import traceback
from typing import List
from langchain_openai import OpenAIEmbeddings

def ensure_openai_base_url_has_v1(url: str) -> str:
    """
    若用户输入的 url 不包含 '/v1'，则在末尾追加 '/v1'。
    """
    import re
    url = url.strip()
    if not url:
        return url
    if not re.search(r'/v\d+$', url):
        if '/v1' not in url:
            url = url.rstrip('/') + '/v1'
    return url

class BaseEmbeddingAdapter:
    """
    Embedding 接口统一基类
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> List[float]:
        raise NotImplementedError

class OpenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    基于 OpenAIEmbeddings（或兼容接口）的适配器
    """
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self._embedding = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=ensure_openai_base_url_has_v1(base_url),
            model=model_name
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedding.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self._embedding.embed_query(query)

class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    其接口路径为 /api/embeddings
    """
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            vec = self._embed_single(text)
            embeddings.append(vec)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(query)

    def _embed_single(self, text: str) -> List[float]:
        """
        调用 Ollama 本地服务 /api/embeddings 接口，获取文本 embedding
        """
        url = self.base_url
        if "api/embeddings" not in url:
            url = f"{url}/api/embeddings"

        data = {
            "model": self.model_name,
            "prompt": text
        }
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            if "embedding" not in result:
                raise ValueError("No 'embedding' field in Ollama response.")
            return result["embedding"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama embeddings request error: {e}\n{traceback.format_exc()}")
            return []

class MLStudioEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self._embedding = OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=ensure_openai_base_url_has_v1(base_url),
            model=model_name
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedding.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self._embedding.embed_query(query)

def create_embedding_adapter(
    interface_format: str,
    api_key: str,
    base_url: str,
    model_name: str
) -> BaseEmbeddingAdapter:
    """
    工厂函数：根据 interface_format 返回不同的 embedding 适配器实例
    """
    if interface_format.lower() == "openai":
        return OpenAIEmbeddingAdapter(api_key, base_url, model_name)
    elif interface_format.lower() == "ollama":
        return OllamaEmbeddingAdapter(model_name, base_url)
    elif interface_format.lower() == "ml studio":
        return MLStudioEmbeddingAdapter(api_key, base_url, model_name)
    else:
        raise ValueError(f"Unknown embedding interface_format: {interface_format}")
