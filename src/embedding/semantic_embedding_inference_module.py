import torch
import transformers
from lightning import LightningModule
from torch import nn

from src.common.components.model_output import ModelOutput
from src.data.components.data_models import ItemBatch


class SemanticEmbeddingInferenceModule(LightningModule):
    def __init__(
        self,
        semantic_embedding_model: nn.Module | transformers.PreTrainedModel,
        semantic_embedding_model_input_map: dict[str, str],
    ):
        """
        Initialize the SemanticEmbeddingInferenceModule.

        This module is used to compute semantic embeddings from input data using a
        pre-trained, frozen semantic embedding model. It is intended to be used only for
        inference.

        Args:
            semantic_embedding_model: The model to use for computing semantic embeddings.
            semantic_embedding_model_input_map: The mapping from feature names to input names
                expected by the semantic embedding model.
        """
        super().__init__()

        self.semantic_embedding_model = semantic_embedding_model
        # We use a frozen embedding module to compute the input embeddings
        for param in self.semantic_embedding_model.parameters():
            param.requires_grad = False
        self.semantic_embedding_model_input_map = semantic_embedding_model_input_map

    def forward(self, model_input: ItemBatch) -> torch.Tensor:
        """
        Get the semantic embeddings from the input data.

        Args:
            model_input: ItemData consisting of the batch of input features.

        Returns:
            semantic_embeddings: The semantic embeddings.
                Shape (batch_size, n_features)
        """
        semantic_embedding_model_arg_to_feature = {
            embedding_model_arg: model_input.features[feature_name]
            for embedding_model_arg, feature_name in self.semantic_embedding_model_input_map.items()
        }
        with torch.no_grad():
            semantic_embeddings = self.semantic_embedding_model(**semantic_embedding_model_arg_to_feature)
        return semantic_embeddings

    def model_step(self, model_input: ItemBatch) -> torch.Tensor:
        semantic_embeddings = self.forward(model_input)
        return semantic_embeddings

    def predict_step(self, batch: ItemBatch) -> ModelOutput:
        """
        Perform a single prediction step on a batch of data.

        Save the semantic embeddings of the input items and the corresponding item ids
        in a ModelOutput object.

        Args:
            batch: A batch of data of ItemData type.

        Returns:
            model_output: A ModelOutput object containing the item
                ids as keys and the semantic embeddings as predictions.
        """
        semantic_embeddings = self.model_step(batch)
        item_ids = [item_id.item() if isinstance(item_id, torch.Tensor) else item_id for item_id in batch.item_ids]

        return ModelOutput(keys=item_ids, predictions=semantic_embeddings)
