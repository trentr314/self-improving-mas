"""Structured output types for agent responses."""

from pydantic import BaseModel, Field


class Proposal(BaseModel):
    """An agent's proposal for improving the system."""
    summary: str = Field(description="One-line summary of the proposal")
    detail: str = Field(description="2-3 sentences explaining the proposal concretely")
    rationale: str = Field(description="Why this would improve the system")


class Critique(BaseModel):
    """An agent's critique of a proposal."""
    target_proposal: int = Field(description="Index of the proposal being critiqued (0-based)")
    weakness: str = Field(description="The single biggest weakness or gap")
    suggested_fix: str = Field(description="A specific, actionable fix for the weakness")


class Refinement(BaseModel):
    """A refined version of a proposal incorporating critique."""
    original_proposal: int = Field(description="Index of the original proposal being refined (0-based)")
    summary: str = Field(description="One-line summary of the refined proposal")
    detail: str = Field(description="2-3 sentences explaining the refined proposal")
    changes_made: str = Field(description="What was changed and why, based on the critique")


class Vote(BaseModel):
    """An agent's vote on a proposal (not their own)."""
    chosen_proposal: int = Field(description="Index of the proposal being voted for (0-based)")
    reasoning: str = Field(description="Why this proposal is the best option")


class ProtocolChange(BaseModel):
    """A proposed change to the coordination protocol."""
    field_path: str = Field(description="Dot-separated path to the field to change, e.g. 'rounds' or 'roles.proposer' or 'phases'")
    new_value: str = Field(description="The new value as a JSON string. For arrays use [\"a\",\"b\"], for objects use {\"key\":\"val\"}, for strings just the string, for numbers the number.")
    summary: str = Field(description="One-line summary of what this change does")
    rationale: str = Field(description="Why this change would improve the system's coordination")
