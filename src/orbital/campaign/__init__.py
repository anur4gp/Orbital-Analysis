"""Monte Carlo campaigns: sweep a parameter space, in parallel or sharded.

The sweep asks one question: over what range of initial uncertainty and
sensor quality does each filter stay consistent? Phase 2 answered it at two
points; this runs a space-filling design over six parameters.
"""
from orbital.campaign.config import PARAMETERS, CampaignConfig, Provenance
from orbital.campaign.design import PointSettings, design_matrix, settings_for
from orbital.campaign.runner import run_campaign, run_point
from orbital.campaign.storage import merge_shards, read_runs, summarize, write_runs

__all__ = [
    "PARAMETERS",
    "CampaignConfig",
    "PointSettings",
    "Provenance",
    "design_matrix",
    "merge_shards",
    "read_runs",
    "run_campaign",
    "run_point",
    "settings_for",
    "summarize",
    "write_runs",
]
