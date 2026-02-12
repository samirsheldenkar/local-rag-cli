"""Custom OpenCLIP embedding model for image and text embeddings."""

from typing import Any, List

import open_clip
import torch
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from PIL import Image


class OpenCLIPEmbedding(MultiModalEmbedding):
    """Embedding model using OpenCLIP for both text and image embeddings.

    This provides a shared embedding space for text and images,
    enabling multimodal RAG retrieval.
    """

    model_name: str = Field(default="ViT-B-32", description="OpenCLIP model name")
    pretrained: str = Field(default="laion2b_s34b_b79k", description="Pretrained weights")

    _model: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, pretrained=pretrained, **kwargs)

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self._device
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()

    @classmethod
    def class_name(cls) -> str:
        return "OpenCLIPEmbedding"

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            text_features = self._model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

    def _get_image_embedding(self, image_path: str) -> List[float]:
        """Embed a single image from a file path."""
        image = Image.open(image_path).convert("RGB")
        image_input = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze().cpu().tolist()

    async def _aget_image_embedding(self, img_file_path: str) -> List[float]:
        return self._get_image_embedding(img_file_path)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Embed a query string (same as text embedding)."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
