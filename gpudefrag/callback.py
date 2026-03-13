"""
gpudefrag.callback — PyTorch training loop integration.

Provides a zero-config callback that hooks into any training loop to
enable automatic predictive defragmentation.

Usage::

    from gpudefrag import DefragCallback

    callback = DefragCallback()
    callback.on_train_begin()

    for epoch in range(num_epochs):
        for batch in dataloader:
            callback.on_step_begin()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            callback.on_step_end()

    callback.on_train_end()
"""

from typing import Optional
from gpudefrag.monitor import DefragMonitor
from gpudefrag.utils import get_logger, DefragConfig

log = get_logger("callback")


class DefragCallback:
    """
    Training loop callback for automatic predictive defragmentation.

    Drop-in compatible with custom training loops and framework callbacks.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        model_path: Optional[str] = None,
        config: Optional[DefragConfig] = None,
    ):
        self.monitor = DefragMonitor(
            model_path=model_path,
            threshold=threshold,
            config=config,
        )
        self._step_count = 0

    def on_train_begin(self) -> None:
        """Called at the start of training."""
        self.monitor.start()
        log.info("DefragCallback activated.")

    def on_train_end(self) -> None:
        """Called at the end of training."""
        stats = self.monitor.stop().stats()
        log.info(
            "Training complete. %d steps, %d compactions, %.1f MB freed.",
            self._step_count,
            stats["total_compactions"],
            stats["total_freed_mb"],
        )

    def on_step_begin(self) -> None:
        """Called before each training step."""
        self.monitor.auto_record()

    def on_step_end(self) -> None:
        """Called after each training step."""
        self.monitor.auto_record()
        self._step_count += 1

    def stats(self) -> dict:
        """Return monitor statistics."""
        return self.monitor.stats()
