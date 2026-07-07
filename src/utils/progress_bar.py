from typing import Any

from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


class StepBasedRichProgressBar(RichProgressBar):
    """Rich progress bar that keeps train progress aligned with global steps."""

    TRAIN_DESCRIPTION = "Train"

    def __init__(
        self,
        refresh_rate: int = 10,
        leave: bool = False,
        theme: RichProgressBarTheme = RichProgressBarTheme(),
        console_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(refresh_rate=refresh_rate, leave=leave, theme=theme, console_kwargs=console_kwargs)

    def _resolve_train_total(self, trainer: Trainer) -> int | float | None:
        max_steps = getattr(trainer, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            return max_steps

        total_train_batches = self.total_train_batches
        max_epochs = getattr(trainer, "max_epochs", None)
        if isinstance(total_train_batches, (int, float)) and isinstance(max_epochs, int) and max_epochs > 0:
            return total_train_batches * max_epochs
        return total_train_batches

    def _resolve_completed_steps(self, trainer: Trainer) -> int:
        completed_steps = trainer.global_step
        total = self._resolve_train_total(trainer)
        if isinstance(total, (int, float)):
            completed_steps = min(completed_steps, int(total))
        return completed_steps

    def _get_train_description(self, current_epoch: int) -> str:  # noqa: ARG002 - epoch intentionally ignored
        return self.TRAIN_DESCRIPTION

    def get_metrics(self, trainer: Trainer, pl_module: Any) -> dict[str, Any]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Any) -> None:
        if self.is_disabled:
            return

        total_steps = self._resolve_train_total(trainer)
        train_description = self._get_train_description(trainer.current_epoch)

        if self.train_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.progress is not None:
            if self.train_progress_bar_id is None:
                self.train_progress_bar_id = self._add_task(total_steps, train_description)
            else:
                self.progress.reset(
                    self.train_progress_bar_id,
                    total=total_steps,
                    completed=self._resolve_completed_steps(trainer),
                    description=f"[{self.theme.description}]{train_description}" if self.theme.description else train_description,
                    visible=True,
                )
        self.refresh()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.is_disabled and self.train_progress_bar_id is None:
            self._initialize_train_progress_bar_id()
            if self.progress is not None and self.train_progress_bar_id is not None:
                self.progress.update(
                    self.train_progress_bar_id,
                    total=self._resolve_train_total(trainer),
                    description=f"[{self.theme.description}]{self.TRAIN_DESCRIPTION}"
                    if self.theme.description
                    else self.TRAIN_DESCRIPTION,
                )

        self._update(self.train_progress_bar_id, self._resolve_completed_steps(trainer))
        self._update_metrics(trainer, pl_module)
        self.refresh()
