"""Data loading and plugin interfaces."""

from ivdyn.data.build_surface import SurfaceDatasetBuildConfig, build_surface_dataset
from ivdyn.data.earnings_extract import EarningsFeatureConfig, daily_earnings_features, load_earnings_calendar
from ivdyn.data.ivyds_extract import WRDSExtractConfig, build_wrds_contract_panel
from ivdyn.data.loaders import DatasetBuildConfig, build_dataset
from ivdyn.data.massive import (
    MassiveFlatfileAggsPlugin,
    MassiveFlatfileMinuteAggsPlugin,
    MassiveRawParquetPlugin,
    MassiveRESTPlugin,
    OptionsDataPlugin,
    PluginFactory,
)

__all__ = [
    "WRDSExtractConfig",
    "build_wrds_contract_panel",
    "SurfaceDatasetBuildConfig",
    "build_surface_dataset",
    "EarningsFeatureConfig",
    "load_earnings_calendar",
    "daily_earnings_features",
    "DatasetBuildConfig",
    "build_dataset",
    "OptionsDataPlugin",
    "PluginFactory",
    "MassiveRawParquetPlugin",
    "MassiveFlatfileAggsPlugin",
    "MassiveFlatfileMinuteAggsPlugin",
    "MassiveRESTPlugin",
]
