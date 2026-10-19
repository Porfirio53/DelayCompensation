"""Named numeric signals; rows are variates and columns are source-grid ticks.

Targets in one view form one joint TiRex-2 series. Covariates condition that
series without becoming forecast outputs. No vehicle-specific scaling lives here.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .data.arrival import _readonly
from .timing.clock import ObservationTiming, integer_us


@dataclass(frozen=True)
class SignalSchema:
    targets: tuple[str, ...]
    past_covariates: tuple[str, ...] = ()
    future_covariates: tuple[str, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for role in ("targets", "past_covariates", "future_covariates"):
            names = getattr(self, role)
            if isinstance(names, str) or not isinstance(names, (list, tuple)):
                raise ValueError(f"{role} must be a list or tuple of signal names")
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError("Signal names must be nonempty strings")
            object.__setattr__(self, role, tuple(names))
        if not self.targets or len(set(self.names)) != len(self.names):
            raise ValueError("Use at least one target and unique names across all signal roles")
        if (not isinstance(self.units, Mapping) or set(self.units) - set(self.names)
                or any(not isinstance(unit, str) for unit in self.units.values())):
            raise ValueError("Units must map declared signal names to strings")
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    @property
    def names(self) -> tuple[str, ...]:
        return self.targets + self.past_covariates + self.future_covariates

    def to_dict(self) -> dict:
        return {"targets": list(self.targets), "past_covariates": list(self.past_covariates),
                "future_covariates": list(self.future_covariates), "units": dict(self.units)}

    def validate_values(self, values: Mapping[str, float]) -> None:
        if not isinstance(values, Mapping) or set(values) != set(self.names):
            raise ValueError(f"A signal packet must contain exactly {list(self.names)}")


@dataclass(frozen=True)
class SignalView:
    """Detached received history. A packet observes all declared rows atomically.

    ``values`` has shape [all schema names, time]. ``known_future`` contains
    explicitly supplied covariates after the latest target tick; NaN marks
    unobserved future positions (TiRex-2, Appendix F). Their historical part
    is already in ``values``. It never contains targets.
    """
    query_time_us: int
    sample_period_us: int
    source_time_us: np.ndarray
    schema: SignalSchema
    values: np.ndarray
    received_mask: np.ndarray
    received_at_us: np.ndarray
    native_source_time_us: np.ndarray
    features: Mapping[str, float]
    horizon_steps: int | None
    ready: bool
    reason: str
    timing: ObservationTiming | None = None
    known_future: np.ndarray | None = None

    @property
    def last_values(self) -> dict[str, float]:
        if not len(self.source_time_us):
            return {}
        return {name: float(self.values[index, -1]) for index, name in enumerate(self.schema.targets)}

    @classmethod
    def from_history(cls, targets: Mapping, *, past_covariates: Mapping | None = None,
                     future_covariates: Mapping | None = None, sample_period_us: int = 25_000,
                     source_time_us: int | None = None, units: Mapping | None = None):
        """Direct SDK input; future covariates include history + known future.

        Future positions may be NaN when only part of the horizon is known.
        Historical positions must be finite; infinity is never a missing value.

        Histories must already be aligned to one regular grid. For delayed or
        lossy live inputs, use CaptureSession to preserve receipt times/masks.
        """
        past_covariates = {} if past_covariates is None else past_covariates
        future_covariates = {} if future_covariates is None else future_covariates
        if any(not isinstance(group, Mapping) for group in (targets, past_covariates, future_covariates)):
            raise ValueError("Targets and covariates must be mappings of names to numeric histories")
        schema = SignalSchema(tuple(targets), tuple(past_covariates), tuple(future_covariates), units or {})
        period = integer_us(sample_period_us, "sample_period_us", 1)
        arrays = [np.asarray(value, dtype=np.float32) for value in targets.values()]
        length = arrays[0].size
        arrays += [np.asarray(value, dtype=np.float32) for value in past_covariates.values()]
        if length < 2 or any(a.shape != (length,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Target and past-covariate histories must be finite 1D arrays of equal length >= 2")
        future = [np.asarray(value, dtype=np.float32) for value in future_covariates.values()]
        if any(a.ndim != 1 or len(a) <= length or not np.isfinite(a[:length]).all()
               or np.isinf(a[length:]).any() for a in future):
            raise ValueError("Future covariates need finite history and future ticks that are finite or NaN")
        if future and len({len(a) for a in future}) != 1:
            raise ValueError("Future-known covariates must have matching lengths")
        arrays += [a[:length] for a in future]
        anchor = (length - 1) * period if source_time_us is None else integer_us(source_time_us, "source_time_us")
        if anchor < (length - 1) * period:
            raise ValueError("History extends before the nonnegative source-clock origin")
        times = anchor - np.arange(length - 1, -1, -1, dtype=np.int64) * period
        return cls(anchor, period, _readonly(times), schema, _readonly(np.stack(arrays)),
                   _readonly(np.ones(length, dtype=bool)), _readonly(times), _readonly(times),
                   MappingProxyType({"context_ready": 1.0, "received_count": float(length)}),
                   0, True, "ok", known_future=_readonly(np.stack([a[length:] for a in future])) if future else None)


def causal_signal_context(view: SignalView) -> np.ndarray:
    """Forward fill lost ticks per row, preserving leading unavailable values."""
    result = view.values.copy()
    last = np.full(result.shape[0], np.nan)
    for index, received in enumerate(view.received_mask):
        if received:
            last = result[:, index].copy()
        else:
            result[:, index] = last
    return result


@dataclass(frozen=True)
class SignalForecast:
    source_time_us: int
    sample_period_us: int
    quantiles: np.ndarray
    values: np.ndarray  # [target, quantile, future step], in the input units.
    target_names: tuple[str, ...]
    status: str = "ok"
    failure_reason: str = ""
    fallback: bool = False
    batch_elapsed_s: float = 0.0
    batch_size: int = 1

    @property
    def prediction_steps(self) -> int:
        return self.values.shape[-1]

    @property
    def future_time_us(self) -> np.ndarray:
        return self.source_time_us + np.arange(1, self.prediction_steps + 1, dtype=np.int64) * self.sample_period_us

    def _median(self) -> np.ndarray:
        indices = np.flatnonzero(np.isclose(self.quantiles, 0.5))
        if indices.size != 1:
            raise ValueError("Trajectory has no unique median quantile")
        return self.values[:, int(indices[0]), :]

    def values_at(self, target_time_us: int) -> dict[str, float]:
        integer_us(target_time_us, "target_time_us")
        step, remainder = divmod(int(target_time_us) - self.source_time_us, self.sample_period_us)
        if remainder or not 1 <= step <= self.prediction_steps:
            raise ValueError("Target must be on the future grid within this trajectory")
        return dict(zip(self.target_names, map(float, self._median()[:, step - 1]), strict=True))

    def values_at_offset(self, source_offset_us: float, last_values: Mapping[str, float]) -> dict[str, float]:
        if (isinstance(source_offset_us, bool) or not np.isfinite(source_offset_us)
                or not 0 <= source_offset_us <= self.prediction_steps * self.sample_period_us + 1e-6):
            raise ValueError("Offset must lie within this trajectory")
        grid = np.arange(self.prediction_steps + 1, dtype=float) * self.sample_period_us
        medians, result = self._median(), {}
        for name, median in zip(self.target_names, medians, strict=True):
            last = last_values[name]
            if not np.isfinite(last):
                raise ValueError("Observed anchors must be finite")
            result[name] = float(np.interp(source_offset_us, grid, np.r_[last, median]))
        return result
