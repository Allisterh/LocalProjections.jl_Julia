# LocalProjections.jl

[![CI](https://github.com/gragusa/LocalProjections.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/gragusa/LocalProjections.jl/actions/workflows/CI.yml) [![codecov.io](http://codecov.io/github/gragusa/LocalProjections.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/LocalProjections.jl?branch=master) [![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826) ![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)

Estimate local projection impulse response functions using horizon-specific linear regressions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/LocalProjections.jl")
```

## Example

```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices, Plots

# Generate sample data
n = 200
shock = randn(n)
y = cumsum(shock) + 0.5 * randn(n)  # y responds to cumulative shocks
df = DataFrame(y = y, shock = shock)

# Estimate local projections: y_{t+h} = α + β * shock_t + ε
# The coefficient β at each horizon h gives the impulse response
lp_result = lp(@formula(leads(y) ~ shock), df; horizon=20)

# Use HAC-robust standard errors (Newey-West); the default is HC1
lp_result = lp_result + vcov(Bartlett{NeweyWest}())

# Extract the impulse response function
irf = coefpath(lp_result; term=:shock)
se = stderror(lp_result; term=:shock)

# Plot IRF with 95% confidence bands
plot(lp_result; term=:shock, levels=[0.95])
```

## License

MIT License - see [LICENSE](LICENSE) for details.
