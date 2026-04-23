# Hybrid LLM Inference - ICML Presentation

This folder contains the Beamer slides for the ICML 2026 submission demo.

## Compilation

### Option 1: Command line
```bash
pdflatex main.tex
pdflatex main.tex  # Run twice for TOC
```

### Option 2: Using Makefile
```bash
make        # Full compile
make quick  # Single pass
make clean  # Remove auxiliary files
make view   # Open PDF
```

### Option 3: Overleaf
Upload `main.tex` to [Overleaf](https://www.overleaf.com/) for online compilation.

## Slide Structure

1. **Motivation & Problem** - LLM pricing landscape, research questions
2. **Offline Algorithms** - Stage 1 (quota) and Stage 2 (quota + concurrency)
3. **Online Algorithms** - Greedy and cost-aware strategies
4. **Experiments** - Results on production traces
5. **Contributions & Future Work** - Summary and next steps

## Requirements

- TeX Live or MiKTeX
- Packages: beamer, tikz, pgfplots, algorithm, algorithmic, booktabs

## Duration

Estimated presentation time: **15-20 minutes** (excluding Q&A)
