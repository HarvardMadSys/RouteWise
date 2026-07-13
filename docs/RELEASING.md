# Releasing RouteWise

RouteWise publishes the dependency-free API-provider wheel through PyPI
Trusted Publishing. The release workflow uses GitHub's short-lived OIDC token;
the repository must not store a long-lived PyPI API token.

## One-time setup

1. Create the GitHub environment `pypi`. Restrict deployments to version tags
   and add required reviewers so the publish job needs explicit approval.
2. On PyPI, configure a trusted publisher (or a pending publisher for the
   first upload) with these exact values:

   - PyPI project: `routewise`
   - GitHub owner: `HarvardMadSys`
   - GitHub repository: `RouteWise`
   - Workflow: `release.yml`
   - Environment: `pypi`

The publish job intentionally has only `id-token: write`; no `PYPI_TOKEN`
secret is read or required. A pending publisher does not reserve the project
name, so configure it close to the first release and publish promptly.

## Release procedure

1. Merge a version-bump PR after the package workflow is green. Update both
   `project.version` in `pyproject.toml` and `routewise.__version__`.
2. From the merged `main`, create and push the matching annotated tag. For
   version `0.2.0`, the only accepted tag is `v0.2.0`:

   ```bash
   git tag -a v0.2.0 -m "RouteWise 0.2.0"
   git push origin v0.2.0
   ```

3. Publish a GitHub Release for that existing tag:

   ```bash
   gh release create v0.2.0 --verify-tag --generate-notes
   ```

Publishing the GitHub Release triggers `.github/workflows/release.yml`. The
workflow verifies the tag/version pair, builds only the narrow wheel, runs the
exact-member checker, installs the artifact outside the checkout, and then
publishes that same artifact to PyPI from the protected `pypi` environment.

PyPI does not allow replacing a file for an existing version. If publication
fails before upload, fix the workflow and rerun the failed job. If PyPI has
already accepted the wheel, prepare a new version instead of reusing the tag.
