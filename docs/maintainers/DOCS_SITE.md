# Publishing the documentation site

The site is built by `.github/workflows/docs.yml` with MkDocs Material and
`mkdocs-static-i18n`. English builds to `/` and Simplified Chinese to `/zh/`.

## Building locally

```bash
uv sync --group docs
uv run mkdocs serve
```

CI runs `mkdocs build --strict` on every pull request. Link validation is on,
so a link to a missing page or a missing anchor fails the build.

Because the English and Chinese API references do not share a heading
structure, a cross-locale anchor link works only when the anchor exists in
both files. `#provider`, `#router`, and `#tuning` do; `#cost-model` is
English-only.

## Turning publishing on

Deployment is opt-in and takes two deliberate steps. Until both are done, the
`deploy` job is skipped and `main` stays green with nothing published.

1. Enable GitHub Pages for the repository with **GitHub Actions** as the
   source, under Settings, Pages.
2. Set the repository variable `PUBLISH_DOCS` to `true`:

   ```bash
   gh variable set PUBLISH_DOCS --body true
   ```

The next push to `main` publishes to `https://harvardmadsys.github.io/RouteWise/`.

Enabling step 1 without step 2 changes nothing. Setting step 2 without step 1
makes the deploy job fail with a 404 from the Pages API.

## Turning publishing off

```bash
gh variable delete PUBLISH_DOCS
```

Removing the variable stops future deployments. It does not unpublish what is
already live; disable Pages under Settings, Pages for that.

## After publishing

Six user-visible links still point at repository paths rather than the site:
four in `README.md` and the `Documentation` project URL in `pyproject.toml`.
Repoint them once the site is reachable, so PyPI and GitHub send readers to
the rendered documentation.
