# Releasing RouteWise

RouteWise publishes the dependency-free API-provider wheel through PyPI
Trusted Publishing. The release workflow uses GitHub's short-lived OIDC token;
the repository must not store a long-lived PyPI API token.

## One-time setup

1. Create the GitHub environment `pypi`. Restrict deployments to version tags
   and add required reviewers so the publish job needs explicit approval.
2. Confirm that the release maintainers can manage the existing `routewise`
   project on PyPI. In that project's **Publishing** settings, add a GitHub
   Actions trusted publisher with these exact values:

   - PyPI project: `routewise`
   - GitHub owner: `HarvardMadSys`
   - GitHub repository: `RouteWise`
   - Workflow: `release.yml`
   - Environment: `pypi`

The project already exists because `0.1.0` through `0.2.0` were uploaded as an
experimental hosted-service SDK. Do not create a pending publisher. The
publish job intentionally has only `id-token: write`; no `PYPI_TOKEN` secret
is read or required.

After the first Trusted Publishing release succeeds, revoke the long-lived
PyPI token used for the manual legacy uploads. Do not delete the legacy
releases: applications may still depend on them, and PyPI version numbers
cannot be reused. If the hosted SDK is no longer safe or supported, yank those
versions with a migration reason after `0.3.0` is available.

## Release procedure

1. Merge a version-bump PR after the package workflow is green. Update both
   `project.version` in `pyproject.toml` and `routewise.__version__`.
2. From the merged `main`, create and push the matching annotated tag. For
   version `0.3.0`, the only accepted tag is `v0.3.0`:

   ```bash
   git tag -a v0.3.0 -m "RouteWise 0.3.0"
   git push origin v0.3.0
   ```

3. Create a draft GitHub Release for that existing tag, review its notes, and
   make the `0.1.x`--`0.2.0` hosted-SDK to `0.3.0` local-library break
   prominent:

   ```bash
   gh release create v0.3.0 --verify-tag --generate-notes --draft
   gh release view v0.3.0 --web
   # After reviewing and editing the notes:
   gh release edit v0.3.0 --draft=false
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
