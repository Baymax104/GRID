from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar


class OneBasedEpochProgressBar(TQDMProgressBar):
    """TQDM progress bar that displays epoch numbers starting at 1.

    This only changes the user-facing description text. Lightning's internal
    ``current_epoch`` and checkpoint semantics remain zero-based.
    """

    def _format_epoch_description(self, trainer: Trainer, prefix: str = "Epoch") -> str:
        current_epoch = trainer.current_epoch + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        if isinstance(max_epochs, int) and max_epochs > 0:
            return f"{prefix} {current_epoch}/{max_epochs}"
        return f"{prefix} {current_epoch}"

    def _set_description(self, bar, trainer: Trainer, prefix: str = "Epoch") -> None:
        if bar is None:
            return

        description = self._format_epoch_description(trainer, prefix=prefix)
        if hasattr(bar, "set_description_str"):
            bar.set_description_str(description)
        else:
            bar.set_description(description)

    def on_train_start(self, trainer: Trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        self._set_description(self.train_progress_bar, trainer)

    def on_train_epoch_start(self, trainer: Trainer, pl_module) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self._set_description(self.train_progress_bar, trainer)

    def on_validation_start(self, trainer: Trainer, pl_module) -> None:
        super().on_validation_start(trainer, pl_module)
        self._set_description(self.val_progress_bar, trainer, prefix="Validation Epoch")

    def on_test_start(self, trainer: Trainer, pl_module) -> None:
        super().on_test_start(trainer, pl_module)
        self._set_description(self.test_progress_bar, trainer, prefix="Test Epoch")

    def on_predict_start(self, trainer: Trainer, pl_module) -> None:
        super().on_predict_start(trainer, pl_module)
        self._set_description(self.predict_progress_bar, trainer, prefix="Predict Epoch")
