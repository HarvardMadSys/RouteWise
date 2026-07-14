# Releasing RouteWise

RouteWise publishes the dependency-free API-provider wheel through PyPI
Trusted Publishing. The release workflow uses GitHub's short-lived OIDC token;
the repository must not store a long-lived PyPI API token.

## One-time setup

1. Create the GitHub environment `pypi`. Restrict deployments to version tags
   and add required reviewers so the publish job needs explicit approval.
2. Resolve control of the `routewise` distribution name. The releases currently
   on PyPI (`0.1.0`--`0.2.0`) belong to an unaffiliated project; they were not
   published by HarvardMadSys. Do not tag or publish a release while the name
   remains outside the team's control.
3. After the project is transferred to a team-controlled PyPI account, add a
   GitHub Actions trusted publisher in its **Publishing** settings with these
   exact values:

   - PyPI project: `routewise`
   - GitHub owner: `HarvardMadSys`
   - GitHub repository: `RouteWise`
   - Workflow: `release.yml`
   - Environment: `pypi`

Do not create a pending publisher for `routewise` while the existing project
occupies that name. The publish job intentionally has only `id-token: write`;
no `PYPI_TOKEN` secret is read or required. Decisions about the unaffiliated
releases (preserve, yank, or remove) must be made with PyPI and the current
owner during transfer; they are not legacy releases of this codebase.

## Release procedure

Do not begin this procedure until the PyPI namespace notice above is resolved
and the trusted publisher is visible in the team-controlled project settings.

1. Merge a version-bump PR after the package workflow is green. Update both
   `project.version` in `pyproject.toml` and `routewise.__version__`.
2. From the merged `main`, create and push the matching annotated tag. For
   version `0.3.0`, the only accepted tag is `v0.3.0`:

   ```bash
   git tag -a v0.3.0 -m "RouteWise 0.3.0"
   git push origin v0.3.0
   ```

3. Create a draft GitHub Release for that existing tag and review its notes.
   Describe `0.3.0` as the first official HarvardMadSys RouteWise package; do
   not describe the unaffiliated PyPI releases as predecessors or a migration:

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
