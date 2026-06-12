from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)


class RestartOnPoorPerformanceCallback(TrainerCallback):
    """
    Only checks performance once at a specific step.
    If performance is poor, triggers restart. Otherwise, never checks again.
    """

    def __init__(
        self, eval_loss_threshold: float, check_at_step: int, max_retries: int = 3
    ):
        """
        Args:
            eval_loss_threshold: Maximum acceptable eval loss at check_at_step
            check_at_step: Exact step at which to check performance (only checks once)
            max_retries: Maximum number of training restarts allowed
        """
        self.eval_loss_threshold = eval_loss_threshold
        self.check_at_step = check_at_step
        self.max_retries = max_retries

        # Internal state
        self.retry_count = 0
        self.should_restart = False
        self.has_checked = False  # Only check once

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Check eval loss only once at the specified step."""

        # Skip if we've already checked
        if self.has_checked:
            return control

        # Only proceed if we have eval loss in the current log
        if state.log_history and "eval_loss" in state.log_history[-1]:
            current_step = state.global_step

            # Check only at the specific step
            if current_step >= self.check_at_step:
                self.has_checked = True  # Mark as checked
                eval_loss = state.log_history[-1]["eval_loss"]

                print(f"\n=== Performance Check at Step {current_step} ===")
                print(
                    f"Eval loss: {eval_loss:.4f}, Threshold: {self.eval_loss_threshold:.4f}"
                )

                # Check if eval loss is above threshold
                if eval_loss > self.eval_loss_threshold:
                    if self.retry_count < self.max_retries:
                        self.should_restart = True
                        control.should_training_stop = True
                        print(
                            f"Poor initial performance. Triggering restart (attempt {self.retry_count + 1}/{self.max_retries})"
                        )
                    else:
                        print(
                            f"Poor performance but max retries ({self.max_retries}) reached. Continuing."
                        )
                else:
                    print(
                        "Good performance! Continuing training without further checks."
                    )

        return control

    def reset_for_retry(self):
        """Reset internal state for a new training attempt."""
        self.retry_count += 1
        self.should_restart = False
        self.has_checked = False
