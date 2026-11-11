"""Interactive chat with the agent in the terminal.

Usage:
    python -m agent.cli                      # the triage chooses the product
    python -m agent.cli --scope novadrive    # product fixed by the caller, no triage
    python -m agent.cli --chat-id demo-1     # resume a chat
"""

import argparse
import asyncio
import json
import uuid
from collections.abc import Callable

from agent import agent_config, runtime, turn_events
from agent.models import chat as chat_models
from shared import kb_index, logger

DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"
EXIT_COMMANDS = {"/exit", "/quit"}
STATE_COMMAND = "/state"
DEMO_USER = "demo-user"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the blueprint agent")
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Fix a scope (repeatable): skips the triage",
    )
    parser.add_argument("--chat-id", help="Id of the chat to start or resume")
    return parser.parse_args()


class TurnPrinter:
    """Prints the events of a turn in the terminal.

    The text of the specialist is printed as it streams. The text of the triage is
    held back until the turn ends: it is printed only if the triage answered (e.g. a
    clarification), and dropped if the triage handed the turn over, because then it
    is a preamble that is not part of the answer.
    """

    def __init__(self, write: Callable[[str], None]):
        """Initializes the printer.

        Args:
            write: Writes a piece of output, without adding a newline.
        """
        self._write = write
        self._held_triage_text: list[str] = []

    def handle(self, event: turn_events.TurnEvent) -> None:
        """Prints one event: text in the default color, tools and handovers dimmed."""
        if isinstance(event, turn_events.TextDelta):
            if event.agent_type == chat_models.TRIAGE_AGENT:
                self._held_triage_text.append(event.text)
            else:
                self._write(event.text)
        elif isinstance(event, turn_events.ToolStarted):
            self._write(f"\n{DIM}  ⚙ {event.tool_name}{RESET}\n")
        elif isinstance(event, turn_events.Handover):
            if event.from_agent == chat_models.TRIAGE_AGENT:
                self._held_triage_text = []
            discarded = " (previous text discarded)" if event.discarded_text else ""
            target = ", ".join(event.scope_ids) if event.scope_ids else "triage"
            self._write(
                f"\n{YELLOW}  ↪ handover {event.from_agent} → {target}{discarded}{RESET}\n"
            )
        elif isinstance(event, turn_events.TurnCompleted):
            # The triage answered the user itself: its held text is the answer
            self._write("".join(self._held_triage_text))
            self._held_triage_text = []
            result = event.result
            self._write(
                f"\n{DIM}  [{result.agent_type} {result.scope_ids} | "
                f"tokens in/out {result.input_tokens}/{result.output_tokens}]{RESET}\n\n"
            )


def _write(text: str) -> None:
    print(text, end="", flush=True)


async def _chat_loop(app: runtime.Runtime, chat_ctx: chat_models.ChatContext) -> None:
    print(
        f"{CYAN}Chat {chat_ctx.chat_id} — type {STATE_COMMAND} for the chat state, /exit to quit{RESET}\n"
    )
    while True:
        try:
            prompt = input(f"{CYAN}you> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt in EXIT_COMMANDS:
            return
        if prompt == STATE_COMMAND:
            print(
                json.dumps(
                    app.chat_store.get_state(chat_ctx.chat_id).model_dump(mode="json"),
                    indent=2,
                )
            )
            continue

        print(f"{CYAN}nova> {RESET}", end="", flush=True)
        printer = TurnPrinter(write=_write)
        async for event in app.orchestrator.run_turn(chat_ctx, prompt):
            printer.handle(event)


def main() -> None:
    """Starts the interactive chat."""
    args = _parse_args()
    config = agent_config.AgentConfig.from_environment()
    log_path = logger.configure_logging(
        logs_dir=config.settings.logs_dir, level=config.settings.log_level
    )
    try:
        app = runtime.build_runtime(config)
    except kb_index.IndexMismatchError as e:
        raise SystemExit(str(e)) from e

    unknown_scope_ids = sorted(set(args.scope) - set(app.scope_catalog.scope_ids()))
    if unknown_scope_ids:
        raise SystemExit(
            f"Unknown scopes {unknown_scope_ids}: use {app.scope_catalog.scope_ids()}"
        )

    chat_ctx = chat_models.ChatContext(
        chat_id=args.chat_id or f"cli-{uuid.uuid4().hex[:8]}",
        user=DEMO_USER,
        scope_ids=args.scope,
    )
    print(f"{DIM}Log: {log_path}{RESET}")
    asyncio.run(_chat_loop(app, chat_ctx))


if __name__ == "__main__":
    main()
