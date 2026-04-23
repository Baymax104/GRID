from typing import Any, Dict, Optional, Tuple, Union

import torch
import transformers
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.aggregation import BaseAggregator

from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.models.components.eval_metrics import Evaluator
from src.models.components.eval_metrics import RetrievalEvaluator
from src.models.components.model_output import SharedKeyAcrossPredictionsOutput
from src.models.modules.embedding_aggregator import EmbeddingAggregator
from src.utils.pylogger import RankedLogger


console_logger = RankedLogger(__name__, rank_zero_only=True)


class TransformerBaseModule(LightningModule):
    def __init__(
        self,
        huggingface_model: transformers.PreTrainedModel,
        postprocessor: torch.nn.Module,
        aggregator: EmbeddingAggregator,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: torch.nn.Module,
        evaluator: Evaluator,
        weight_tying: bool,
        compile: bool,
        training_loop_function: callable = None,
        feature_to_model_input_map: Dict[str, str] = None,
        decoder: torch.nn.Module = None,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # we remove the nn.Modules as they are already checkpointed to avoid doing it twice

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "huggingface_model",
                "postprocessor",
                "aggregator",
                "decoder",
                "loss_function",
            ],
        )

        self.model = huggingface_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.evaluator = evaluator
        self.training_loop_function = training_loop_function
        # We use setters to set the prediction key and name.
        self._prediction_key_name = None
        self._prediction_name = None

        if self.training_loop_function is not None:
            self.automatic_optimization = False

        if self.evaluator:  # For inference, evaluator is not set.
            for metric_name, metric_object in self.evaluator.metrics.items():
                setattr(self, metric_name, metric_object)

            # for averaging loss across batches
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()
            self.test_loss = MeanMetric()

        self.encoder = huggingface_model
        self.embedding_post_processor = postprocessor
        self.decoder = decoder
        self.aggregator = aggregator
        self.feature_to_model_input_map = feature_to_model_input_map if feature_to_model_input_map else {}

    @property
    def prediction_key_name(self) -> Optional[str]:
        return self._prediction_key_name

    @prediction_key_name.setter
    def prediction_key_name(self, value: str) -> None:
        console_logger.debug(f"Setting prediction_key_name to {value}")
        self._prediction_key_name = value

    @property
    def prediction_name(self) -> Optional[str]:
        return self._prediction_name

    @prediction_name.setter
    def prediction_name(self, value: str) -> None:
        console_logger.debug(f"Setting prediction_name to {value}")
        self._prediction_name = value

    def forward(
        self,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def model_step(
        self,
        model_input: Any,
        label_data: Optional[Any] = None,
    ):
        raise NotImplementedError(
            "Inherit from this class and implement the model_step method."
        )

    def get_embedding_table(self):
        if self.hparams.weight_tying:  # type: ignore
            return self.encoder.get_input_embeddings().weight
        else:
            return self.decoder.weight

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        if self.evaluator:
            self.val_loss.reset()
            self.evaluator.reset()
            self.train_loss.reset()
            self.test_loss.reset()

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        if self.evaluator:
            self.val_loss.reset()
            self.evaluator.reset()

    def on_test_epoch_start(self):
        if self.evaluator:
            self.test_loss.reset()
            self.evaluator.reset()

    def on_validation_epoch_end(self) -> None:
        # Lightning hook that is called when a validation epoch ends.
        if self.evaluator:
            self.log("val/loss", self.val_loss, sync_dist=False, prog_bar=True, logger=True)
            self.log_metrics("val")

    def on_test_epoch_end(self) -> None:
        if self.evaluator:
            self.log("test/loss", self.test_loss, sync_dist=False, prog_bar=True, logger=True)
            self.log_metrics("test")

    def on_exception(self, exception):
        self.trainer.should_stop = True  # stop all workers
        if self.trainer.logger is not None:
            self.trainer.logger.finalize(status="failure")

    def log_metrics(
        self,
        prefix: str,
        on_step=False,
        on_epoch=True,
        # We use sync_dist=False by default because, if using retrieval metrics, those are already synchronized. Change if using
        # different metrics than the default ones.
        sync_dist=False,
        logger=True,
        prog_bar=False,
        call_compute=False,
    ):

        metrics_dict = {
            f"{prefix}/{metric_name}": metric_object.compute()
            if call_compute
            else metric_object
            for metric_name, metric_object in self.evaluator.metrics.items()
        }

        self.log_dict(
            metrics_dict,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            logger=logger,
            prog_bar=prog_bar,
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def training_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (tuple). Because of lightning, the tuple is wrapped in another tuple,
        and the actual batch is at position 0. The batch is a tuple of data where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        # Batch will be a tuple of model inputs and labels. We use the index here to access them.
        model_output, loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        # update and log metrics. Will only be logged at the interval specified in the logger config
        self.train_loss(loss)
        # checks logging interval and logs the loss
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # If a training loop function is passed, we call it with the module and the loss.
        # otherwise we use the automatic optimization provided by lightning
        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        return loss

    def eval_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        loss_to_aggregate: BaseAggregator,
    ):
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]

        model_output_before_aggregation, loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        model_output_after_aggregation = self.aggregator(
            model_output_before_aggregation, model_input.mask
        )

        # Updates metrics inside evaluator.
        self.evaluator(
            query_embeddings=model_output_after_aggregation,
            key_embeddings=self.get_embedding_table().to(model_output_after_aggregation.device),
            # TODO: (lneves) hardcoded for now, will need to change for multiple features
            labels=list(label_data.labels.values())[0].to(model_output_after_aggregation.device),
        )
        loss_to_aggregate(loss)

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.val_loss)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.test_loss)

    def predict_step(
        self,
        batch: Tuple[SequentialModelInputData, SequentialModuleLabelData],
        batch_idx: int,
    ):
        """
        Perform a single prediction step on a batch of data from the test set.

        :param

        Args:
            batch: A batch of data (tuple) where first object is a SequentialModelInputData object
                and second is a SequentialModuleLabelData object.
            batch_idx: batch index
        """
        model_input: SequentialModelInputData = batch[0]
        model_output_before_aggregation, _ = self.model_step(model_input=model_input)

        model_output_after_aggregation = self.aggregator(
            model_output_before_aggregation, model_input.mask
        )
        model_output = SharedKeyAcrossPredictionsOutput(
            key=batch_idx,
            predictions=model_output_after_aggregation,
            key_name=self.prediction_key_name,
            prediction_name=self.prediction_name,
        )
        return model_output
