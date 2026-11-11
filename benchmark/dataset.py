"""Benchmark datasets: scripted conversations with the expected routing and answer."""

import pathlib

import pydantic


class Scenario(pydantic.BaseModel):
    """One test case.

    Attributes:
        scenario_id: Unique id of the case.
        title: Short description.
        scope_ids: Scopes fixed by the caller; empty to go through the triage.
        turns: User messages, sent in order (scripted multi-turn).
        gold_scopes: Scopes the triage should choose at the first turn; empty when
            it should ask a clarification. None when routing is not evaluated.
        ground_truth: Facts a correct final answer contains; empty to skip the quality judges.
        must_include: Behaviours the final answer must have.
        must_not_include: Behaviours or contents the final answer must not have.
        should_open_ticket: Whether the agent should open a ticket during the conversation.
        expected_tools: Tool names (or name prefixes) expected in the conversation, informational.
    """

    scenario_id: str = pydantic.Field(min_length=1)
    title: str
    scope_ids: list[str] = pydantic.Field(default_factory=list)
    turns: list[str] = pydantic.Field(min_length=1)
    gold_scopes: list[str] | None = None
    ground_truth: list[str] = pydantic.Field(default_factory=list)
    must_include: list[str] = pydantic.Field(default_factory=list)
    must_not_include: list[str] = pydantic.Field(default_factory=list)
    should_open_ticket: bool = False
    expected_tools: list[str] = pydantic.Field(default_factory=list)


class Dataset(pydantic.BaseModel):
    """A named list of scenarios."""

    dataset_name: str
    scenarios: list[Scenario] = pydantic.Field(min_length=1)

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "Dataset":
        """Loads and validates a dataset file.

        Raises:
            pydantic.ValidationError: If the file does not match the schema.
        """
        dataset = cls.model_validate_json(path.read_text(encoding="utf-8"))
        scenario_ids = [scenario.scenario_id for scenario in dataset.scenarios]
        duplicated_ids = sorted(
            {
                scenario_id
                for scenario_id in scenario_ids
                if scenario_ids.count(scenario_id) > 1
            }
        )
        if duplicated_ids:
            raise ValueError(f"Duplicated scenario ids in {path}: {duplicated_ids}")
        return dataset
