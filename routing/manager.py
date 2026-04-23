from __future__ import annotations

from typing import TYPE_CHECKING

from .config import RoutingConfig, load_routing_config
from .health import HealthMonitor
from .strategies import FixedRatioStrategy

if TYPE_CHECKING:
    from pathlib import Path

    from serving.adapters.base import BaseAdapter

    from .executor import RouteExecutor


class RoutingManager:
    """Decision layer for applying routing weights to the executor.

    This manager loads configuration, optionally starts health monitoring,
    and computes per-adapter weights that are applied to the `RouteExecutor`.
    """

    def __init__(self, router: RouteExecutor, config_path: Path) -> None:
        self.router = router
        self.config_path = config_path
        self.conf: RoutingConfig | None = None
        self.health: HealthMonitor | None = None

    def load(self) -> None:
        """Load routing configuration and start health monitoring if enabled."""
        self.conf = load_routing_config(self.config_path)
        if self.conf.health_check > 0:
            # Only monitor local deployments. Remote provider endpoints generally do not
            # expose our /health path and would be falsely marked unhealthy.
            endpoints = [d.endpoint for d in self.conf.local_deployment]
            self.health = HealthMonitor(
                timeout_s=self.conf.timeout, interval_s=self.conf.health_check
            )
            self.health.start(endpoints)

    async def shutdown(self) -> None:
        """Shutdown health monitor if running."""
        if self.health:
            await self.health.shutdown()

    def _group_adapters(
        self,
    ) -> dict[str, tuple[list[tuple[BaseAdapter, str]], list[tuple[BaseAdapter, str]]]]:
        """Return mapping model_id -> (local_adapters, remote_adapters).

        Each entry is a list of (adapter, model_id) pairs.
        """
        assert self.conf is not None
        local_eps = {d.endpoint for d in self.conf.local_deployment}
        remote_eps = {d.endpoint for d in self.conf.remote_deployment}
        local_models: set[str] = {m for d in self.conf.local_deployment for m in d.models}
        remote_models: set[str] = {m for d in self.conf.remote_deployment for m in d.models}

        groups: dict[str, tuple[list[tuple[BaseAdapter, str]], list[tuple[BaseAdapter, str]]]] = {}
        for model_id, route_cfg in self.router.routes.items():
            local: list[tuple[BaseAdapter, str]] = []
            remote: list[tuple[BaseAdapter, str]] = []
            for adapter, _w in route_cfg.adapters:
                ep = adapter.config.base_url.rstrip("/")
                # Select by endpoint group and model listing. Exact match only.
                if ep in local_eps and (model_id in local_models):
                    # Optionally skip unhealthy endpoints
                    if not self.health or self.health.is_healthy(ep):
                        local.append((adapter, model_id))
                elif (
                    ep in remote_eps
                    and (model_id in remote_models)
                    and (not self.health or self.health.is_healthy(ep))
                ):
                    remote.append((adapter, model_id))
            groups[model_id] = (local, remote)
        return groups

    def apply(self) -> int:
        """Compute and apply weights to RouteExecutor routes.

        Returns the number of models whose routes were updated.
        """
        assert self.conf is not None
        if self.conf.routing_strategy != "fixed":
            # For now only fixed supported; ignore otherwise
            return 0
        strat = FixedRatioStrategy(local_fraction=self.conf.routing_parameter.local_fraction)
        groups = self._group_adapters()
        updated = 0
        for model_id, route_cfg in list(self.router.routes.items()):
            local, remote = groups.get(model_id, ([], []))
            if not local and not remote:
                # Nothing to override for this model
                continue
            weights = strat.assign(local, remote)
            # Build new adapters list preserving order: first those with new weights, else keep existing
            new_adapters: list[tuple[BaseAdapter, float]] = []
            seen: set[BaseAdapter] = set()
            for adapter, _ in route_cfg.adapters:
                if adapter in weights:
                    new_adapters.append((adapter, weights[adapter]))
                    seen.add(adapter)
            # Include any remaining adapters not covered by config with their old weights
            for adapter, w in route_cfg.adapters:
                if adapter not in seen and adapter not in weights:
                    new_adapters.append((adapter, w))
            if new_adapters:
                self.router.routes[model_id] = type(route_cfg)(adapters=new_adapters)  # type: ignore
                updated += 1
        return updated

    def get_status(self) -> dict[str, object]:
        """Return a summary of current routing status for diagnostics.

        Returns:
            A dictionary with strategy, health settings, and deployment counts.
        """
        conf = self.conf
        if not conf:
            return {"loaded": False}
        return {
            "loaded": True,
            "strategy": conf.routing_strategy,
            "health_check_interval": conf.health_check,
            "timeout": conf.timeout,
            "deployments": {
                "local": len(conf.local_deployment),
                "remote": len(conf.remote_deployment),
            },
        }
