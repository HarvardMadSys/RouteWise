from types import SimpleNamespace

from scripts.shared_profile_probe import _filter_probe_providers


def test_filter_probe_providers_excludes_named_provider() -> None:
    providers = [
        SimpleNamespace(name="Featherless_SC"),
        SimpleNamespace(name="OR_WandB"),
    ]

    filtered = _filter_probe_providers(providers, {"Featherless_SC"})

    assert [provider.name for provider in filtered] == ["OR_WandB"]
