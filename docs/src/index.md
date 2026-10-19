# LocalProjections.jl

A Julia package for estimating Local Projection Impulse Response Functions.

## Features

- Formula-based interface using StatsModels.jl
- Forward-looking responses with `leads()`
- Cumulative responses with `cumul()`
- Anchored responses with `anchor()` or pipe syntax `|`
- Robust inference via CovarianceMatrices.jl (HC, HAC, cluster-robust)
- Plot recipes for IRF visualization
- Tabular summaries with `summarize()`

## Installation

```julia
using Pkg
Pkg.add("LocalProjections")
```

## Quick Start

```julia
using LocalProjections, DataFrames, CovarianceMatrices

# Create data
df = DataFrame(y = randn(100), x = randn(100))

# Estimate local projections
lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)

# Get impulse response
irf = coefpath(lp_result)

# Standard errors (HC1 by default; HAC via `+ vcov(...)`)
se = stderror(lp_result)
lp_hac = lp_result + vcov(Bartlett{NeweyWest}())

# Summary table
summarize(lp_hac)
```

## Tutorials

- [Getting Started](tutorials/basic.md) - Basic local projections without transformations
- [Transformations](tutorials/transformations.md) - Lags, leads, and cumulative responses
- [Inference and Plotting](tutorials/inference.md) - Robust standard errors and visualization
