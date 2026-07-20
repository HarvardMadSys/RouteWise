# Releasing RouteWise

RouteWise publishes the dependency-free API-provider wheel through PyPI
Trusted Publishing. The release workflow uses GitHub's short-lived OIDC token;
the repository must not store a long-lived PyPI API token.

## One-time setup

1. Create the GitHub environment `pypi`. Restrict deployments to version tags
   and add required reviewers so the publish job needs explicit approval.
2. Create a pending GitHub Actions trusted publisher for the unclaimed
   `llm-routewise` project in PyPI's **Publishing** settings with these exact
   values:

   - PyPI project: `llm-routewise`
   - GitHub owner: `HarvardMadSys`
   - GitHub repository: `RouteWise`
   - Workflow: `release.yml`
   - Environment: `pypi`

The unrelated `routewise` project remains outside the team's control and must
not be used for this release. The publish job intentionally has only
`id-token: write`; no `PYPI_TOKEN` secret is read or required. The first
successful trusted publication creates the `llm-routewise` project.

## Release procedure

Do not begin this procedure until the `llm-routewise` pending publisher and the
protected `pypi` environment are configured.

1. Merge a version-bump PR after the package workflow is green. Update both
   `project.version` in `pyproject.toml` and `llm_routewise.__version__`.
2. From the merged `main`, create and push the matching annotated tag. For
   version `0.1.0`, the only accepted tag is `v0.1.0`:

   ```bash
   git tag -a v0.1.0 -m "RouteWise 0.1.0"
   git push origin v0.1.0
   ```

3. Create a draft GitHub Release for that existing tag and review its notes.
   Describe `0.1.0` as the first official HarvardMadSys RouteWise package under
   the `llm-routewise` distribution; do not describe the unaffiliated
   `routewise` releases as predecessors or a migration:

   ```bash
   gh release create v0.1.0 --verify-tag --generate-notes --draft
   gh release view v0.1.0 --web
   # After reviewing and editing the notes:
   gh release edit v0.1.0 --draft=false
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
