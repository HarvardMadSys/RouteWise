# Legacy

Temporary compatibility area for old implementations during migration.

Rule:

```text
legacy/ is kill-on-reproduce.
```

Move code here only after the new `rwsim/` path can reproduce the required
golden baselines or paper experiment outputs. Once reproduced, delete the
legacy path instead of extending it.
