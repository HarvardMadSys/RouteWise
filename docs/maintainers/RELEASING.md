# Releasing RouteWise

This maintainer guide covers publishing the dependency-free API-provider wheel
through PyPI Trusted Publishing. The workflow uses GitHub's short-lived OIDC token;
the repository must not store a long-lived PyPI API token.

## One-time setup

1. Create the GitHub environment `pypi`. Restrict deployments to version tags
   and add required reviewers so the publish job needs explicit approval.
2. Configure a GitHub Actions trusted publisher for the `llm-routewise`
   project in PyPI's **Publishing** settings with these exact values:

   - PyPI project: `llm-routewise`
   - GitHub owner: `HarvardMadSys`
   - GitHub repository: `RouteWise`
   - Workflow: `release.yml`
   - Environment: `pypi`

The unrelated `routewise` project remains outside the team's control and must
not be used for a release. The publish job intentionally has only
`id-token: write`; no `PYPI_TOKEN` secret is read or required. For a new PyPI
project, configure the same values as a pending trusted publisher.

## Release procedure

Do not begin this procedure until the `llm-routewise` trusted publisher and the
protected `pypi` environment are configured.

1. Merge a version-bump PR after the package workflow is green. Update both
   `project.version` in `pyproject.toml` and `llm_routewise.__version__`.
2. From the merged `main`, create and push the matching annotated tag. For
   a release version `X.Y.Z`, use the tag `vX.Y.Z`:

   ```bash
   git tag -a vX.Y.Z -m "RouteWise X.Y.Z"
   git push origin vX.Y.Z
   ```

3. Create a draft GitHub Release for that existing tag and review its notes.
   Describe only changes to the HarvardMadSys `llm-routewise` distribution; do
   not describe the unaffiliated `routewise` releases as predecessors or a
   migration:

   ```bash
   gh release create vX.Y.Z --verify-tag --generate-notes --draft
   gh release view vX.Y.Z --web
   # After reviewing and editing the notes:
   gh release edit vX.Y.Z --draft=false
   ```

Publishing the GitHub Release triggers `.github/workflows/release.yml`. The
workflow verifies the tag/version pair, builds only the narrow wheel, runs the
exact-member and metadata checks, confirms that the tag belongs to `main` and
that the version is unused on PyPI, installs the artifact outside the
checkout, and then publishes that same artifact to PyPI from the protected
`pypi` environment.

PyPI does not allow replacing a file for an existing version. If publication
fails before upload, fix the workflow and rerun the failed job. If PyPI has
already accepted the wheel, prepare a new version instead of reusing the tag.
