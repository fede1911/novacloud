"""Hook that retries a model call whose response has neither text nor tool calls."""

from strands import hooks

from shared import logger

_LOGGER = logger.get_logger(__name__)
MAX_EMPTY_RESPONSE_RETRIES = 2


class EmptyResponseRetry(hooks.HookProvider):
    """Retries the model call when the response has no text and no tool call.

    An empty assistant message would leave the history with two user messages
    in a row. A discarded response triggered no tool, so retrying is safe.
    """

    def __init__(self) -> None:
        """Initializes the counter of the consecutive empty responses."""
        self._consecutive_empty_responses = 0

    def register_hooks(self, registry: hooks.HookRegistry) -> None:
        """Registers the check after every model call."""
        registry.add_callback(hooks.AfterModelCallEvent, self.retry_on_empty_response)

    async def retry_on_empty_response(self, event: hooks.AfterModelCallEvent) -> None:
        """Asks for a retry if the response is empty, at most MAX_EMPTY_RESPONSE_RETRIES times in a row."""
        if event.stop_response is None:
            # Failed invocation: handled by ModelFallbackRetry
            return

        if self.has_usable_content(event.stop_response.message):
            self._consecutive_empty_responses = 0
            return

        if self._consecutive_empty_responses >= MAX_EMPTY_RESPONSE_RETRIES:
            _LOGGER.error(
                "Empty model response %s times in a row: giving up",
                self._consecutive_empty_responses + 1,
            )
            self._consecutive_empty_responses = 0
            return

        self._consecutive_empty_responses += 1
        _LOGGER.warning(
            "Empty model response: retrying (%s/%s)",
            self._consecutive_empty_responses,
            MAX_EMPTY_RESPONSE_RETRIES,
        )
        event.retry = True

    @staticmethod
    def has_usable_content(message: dict) -> bool:
        """True if the message has some text or at least one tool call."""
        return any(
            "toolUse" in block or str(block.get("text", "")).strip()
            for block in message.get("content", [])
        )
