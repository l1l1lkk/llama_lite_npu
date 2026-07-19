"""Canonical serving benchmark harness (schema version 2)."""

from .schema import CampaignSpec, CaseSpec, load_campaign

__all__ = ["CampaignSpec", "CaseSpec", "load_campaign"]
