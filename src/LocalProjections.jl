module LocalProjections

export AbstractLocalProjection, LocalProjection, LocalProjectionIV, IRFSummary
export lp, lpiv, TSLS, horizonmodel, coefpath, nobs, stderror, vcov, summarize, first_stage, weakivtest
export WeakIVTestResult, FirstStageIV
export lag, lead, cumul, CumulTerm, lags, leads, LeadTerm, anchor, AnchorTerm
export as_irf_result, LocalProjectionIRFResult
export irfplot, irfplot!

using PrettyTables: pretty_table, TextHighlighter, TextTableFormat,
    text_table_borders__unicode_rounded, @crayon_str
using Tables: Tables
using StatsModels: StatsModels, AbstractTerm, Term, FunctionTerm, ConstantTerm,
    FormulaTerm, term
using Regress: Regress, OLSMatrixEstimator, IVMatrixEstimator, ols, iv, TSLS,
    AbstractIVEstimator,
    VcovSpec, WeakIVTestResult, FirstStageIV, lags, first_stage, weakivtest
using CovarianceMatrices: CovarianceMatrices
using Statistics: quantile
using Distributions: Normal
using RecipesBase: RecipesBase, @recipe
using ShiftedArrays: lag, lead
using StatsBase: StatsBase, StatisticalModel, coef, nobs, response, stderror, vcov
using AxisArrays: AxisArrays, AxisArray, Axis
using MacroEconometricTools: MacroEconometricTools, LocalProjectionIRFResult, irfplot, irfplot!,
    point_estimate, lowerbounds, upperbounds, coverages

# ============================================================================
# Helper Functions for Common Patterns
# ============================================================================

"""
    _parse_unary_binary_args(t::FunctionTerm, func_name::String, default_value)

Parse arguments from a FunctionTerm that accepts 1 or 2 arguments.
Returns (term, param_value) where param_value is either the provided value or default_value.
"""
function _parse_unary_binary_args(t::FunctionTerm, func_name::String, default_value)
    if length(t.args) == 1
        return (first(t.args), default_value)
    elseif length(t.args) == 2
        term, param_arg = t.args
        (param_arg isa ConstantTerm) ||
            throw(ArgumentError("$func_name parameter must be a number (got $param_arg)"))
        return (term, param_arg.n)
    else
        throw(ArgumentError("$func_name() requires 1 or 2 arguments"))
    end
end

"""
    _extract_single_column(cols, term_name::String="term")

Extract a single column from a matrix or vector, throwing an error if multiple columns.
Returns a Vector (using vec() for matrices).
"""
function _extract_single_column(cols, term_name::String = "term")
    if cols isa AbstractMatrix
        size(cols, 2) == 1 ||
            throw(ArgumentError("$term_name must be a single variable, got $(size(cols, 2)) columns"))
        return vec(cols)
    end
    return cols
end

"""
    _check_horizon_provided(horizon::Union{Int,Nothing}, func_name::String)

Check that horizon is not nothing (for standalone formula usage).
Throws error if horizon is nothing.
"""
function _check_horizon_provided(horizon::Union{Int, Nothing}, func_name::String)
    horizon === nothing &&
        throw(ArgumentError("$func_name() without explicit horizon can only be used in lp() context"))
    return horizon
end

"""
    _termvars_unary(t::FunctionTerm)

Extract termvars from a unary function term (extracts from first argument).
"""
function _termvars_unary(t::FunctionTerm)
    length(t.args) >= 1 && return StatsModels.termvars(t.args[1])
    return Symbol[]
end

"""
    _unwrap_lhs(lhs_term)

Unwrap LHS term to determine if it's anchored, cumulative, or leads.
Returns (is_anchor, is_cumul, is_leads, anchor_term, cumul_term, leads_term).
"""
function _unwrap_lhs(lhs_term)
    if lhs_term isa AnchorTerm
        inner = lhs_term.response
        is_cumul = inner isa CumulTerm
        is_leads = inner isa LeadTerm || !is_cumul  # Default to leads
        return (
            true, is_cumul, is_leads, lhs_term,
            is_cumul ? inner : nothing,
            is_leads && inner isa LeadTerm ? inner : nothing,
        )
    else
        return (
            false, lhs_term isa CumulTerm, lhs_term isa LeadTerm,
            nothing, lhs_term isa CumulTerm ? lhs_term : nothing,
            lhs_term isa LeadTerm ? lhs_term : nothing,
        )
    end
end

"""
    _extract_single_response(term, context::String)

Extract a single response variable from a term, throwing an error if multiple variables.
Returns the Symbol of the single base variable.
"""
function _extract_single_response(term, context::String)::Symbol
    vars = _extract_base_variables(term)
    length(vars) == 1 ||
        throw(ArgumentError("$context must reference a single base variable"))
    return vars[1]
end

"""
    _build_lhs_for_horizon(h::Int, is_anchor, is_cumul, is_leads, anchor_term, cumul_term, leads_term)

Build the LHS term for a specific horizon h, handling anchor/cumul/leads combinations.
"""
function _build_lhs_for_horizon(
        h::Int, is_anchor, is_cumul, is_leads,
        anchor_term, cumul_term, leads_term
    )
    if is_anchor
        inner = if is_cumul && cumul_term !== nothing
            CumulTerm{typeof(cumul_term.term)}(cumul_term.term, h)
        elseif is_leads && leads_term !== nothing
            LeadTerm{typeof(leads_term.term)}(leads_term.term, h)
        else
            LeadTerm{typeof(anchor_term.response)}(anchor_term.response, h)
        end
        return AnchorTerm{typeof(inner), typeof(anchor_term.anchor)}(
            inner, anchor_term.anchor, 0
        )
    elseif is_cumul
        return CumulTerm{typeof(cumul_term.term)}(cumul_term.term, h)
    elseif is_leads
        return LeadTerm{typeof(leads_term.term)}(leads_term.term, h)
    else
        throw(ArgumentError("Invalid LHS term type"))
    end
end

# ============================================================================
# Cumulative Sum Term (cumul)
# This implements cumul(y) for cumulative impulse responses
# Supports nested transforms like cumul(log(y))
# ============================================================================

"""
    cumul(term)
    cumul(term, horizon)

Create cumulative sum term. Used in formulas like:
- `@formula(cumul(y) ~ x)` - horizon determined by lp() context
- `@formula(cumul(y, 5) ~ x)` - explicit horizon for standalone use
- `@formula(cumul(log(y)) ~ x)` - supports nested transformations
"""
cumul(t::T) where {T <: AbstractTerm} = CumulTerm{T}(t, nothing)
cumul(t::T, h::Int) where {T <: AbstractTerm} = CumulTerm{T}(t, h)

# termvars: Extract variables from cumul() for schema creation
StatsModels.termvars(t::FunctionTerm{typeof(cumul)}) = _termvars_unary(t)

# Struct for cumulative sum term
struct CumulTerm{T <: AbstractTerm} <: AbstractTerm
    term::T                      # The term to cumulate (can be nested like log(y))
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::CumulTerm) = (t.term,)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(cumul)}, sch::StatsModels.Schema, ctx::Type
    )
    term, horizon = _parse_unary_binary_args(t, "cumul", nothing)
    term = StatsModels.apply_schema(term, sch, ctx)
    return CumulTerm{typeof(term)}(term, horizon)
end

function StatsModels.apply_schema(t::CumulTerm, sch::StatsModels.Schema, ctx::Type)
    term = StatsModels.apply_schema(t.term, sch, ctx)
    return CumulTerm{typeof(term)}(term, t.horizon)
end

# modelcols: Apply cumulative sum transformation
# Note: In lp() context, horizon is nothing and will be handled specially
function StatsModels.modelcols(ct::CumulTerm, d::Tables.ColumnTable)
    _check_horizon_provided(ct.horizon, "cumul")
    original_cols = StatsModels.modelcols(ct.term, d)
    original_cols = _extract_single_column(original_cols, "cumul() response")
    return _create_cumulative(original_cols, ct.horizon)
end

# width: Return number of columns (always 1 for cumul)
StatsModels.width(ct::CumulTerm) = 1

# show: Display the term
function Base.show(io::IO, ct::CumulTerm)
    return if ct.horizon === nothing
        print(io, "cumul($(ct.term))")
    else
        print(io, "cumul($(ct.term), $(ct.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(ct::CumulTerm)
    base_names = StatsModels.coefnames(ct.term)
    if ct.horizon === nothing
        return ["cumul(" * base_names[1] * ")"]
    else
        return ["cumul(" * base_names[1] * ", $(ct.horizon))"]
    end
end

# ============================================================================
# Lead Term (leads)
# This implements leads(y) for forward-looking regressions with NaN handling
# Named 'leads' to avoid conflict with ShiftedArrays.lead
# Uses our _lead_to_float64() for type stability (NaN instead of missing)
# ============================================================================

"""
    leads(term)
    leads(term, horizon)

Create lead term with NaN handling. Used in formulas like:
- `@formula(leads(y) ~ x)` - horizon determined by lp() context
- `@formula(leads(y, 3) ~ x)` - explicit horizon for standalone use
- `@formula(leads(log(y)) ~ x)` - supports nested transformations

Note: Named 'leads' to distinguish from ShiftedArrays.lead (which returns missing).
This version returns Float64 with NaN for type stability.
"""
leads(t::T) where {T <: AbstractTerm} = LeadTerm{T}(t, nothing)
leads(t::T, h::Int) where {T <: AbstractTerm} = LeadTerm{T}(t, h)

# termvars: Extract variables from leads() for schema creation
StatsModels.termvars(t::FunctionTerm{typeof(leads)}) = _termvars_unary(t)

# Struct for lead term
struct LeadTerm{T <: AbstractTerm} <: AbstractTerm
    term::T                      # The term to lead (can be nested like log(y))
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::LeadTerm) = (t.term,)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(leads)}, sch::StatsModels.Schema, ctx::Type
    )
    term, horizon = _parse_unary_binary_args(t, "leads", nothing)
    term = StatsModels.apply_schema(term, sch, ctx)
    return LeadTerm{typeof(term)}(term, horizon)
end

function StatsModels.apply_schema(t::LeadTerm, sch::StatsModels.Schema, ctx::Type)
    term = StatsModels.apply_schema(t.term, sch, ctx)
    return LeadTerm{typeof(term)}(term, t.horizon)
end

# modelcols: Apply lead transformation with NaN handling
function StatsModels.modelcols(lt::LeadTerm, d::Tables.ColumnTable)
    _check_horizon_provided(lt.horizon, "leads")
    original_cols = StatsModels.modelcols(lt.term, d)
    original_cols = _extract_single_column(original_cols, "leads() response")
    return lead(original_cols, lt.horizon, default = NaN)
end

# width: Return number of columns (always 1 for leads)
StatsModels.width(lt::LeadTerm) = 1

# show: Display the term
function Base.show(io::IO, lt::LeadTerm)
    return if lt.horizon === nothing
        print(io, "leads($(lt.term))")
    else
        print(io, "leads($(lt.term), $(lt.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(lt::LeadTerm)
    base_names = StatsModels.coefnames(lt.term)
    if lt.horizon === nothing
        return ["leads(" * base_names[1] * ")"]
    else
        return ["leads(" * base_names[1] * ", $(lt.horizon))"]
    end
end

# ============================================================================
# Anchor Term (anchor)
# This implements anchor(y, z) for anchored responses: y_{t+h} - z_t
# The anchor variable z stays fixed at time t while y shifts forward to t+h
# ============================================================================

"""
    anchor(response_term, anchor_term)
    anchor(response_term, anchor_term, horizon)

Create anchored response term for local projections. Used in formulas like:
- `@formula(anchor(y, z) ~ x)` - horizon determined by lp() context
- `@formula(anchor(y, z, 5) ~ x)` - explicit horizon for standalone use
- `@formula(anchor(log(y), z) ~ x)` - supports nested transformations on response

Computes y_{t+h} - z_t where:
- y is the response variable (can be transformed)
- z is the anchor variable (stays at time t)
- h is the horizon (0, 1, 2, ...)

At h=0: returns y_t - z_t
At h=1: returns y_{t+1} - z_t
At h=2: returns y_{t+2} - z_t, etc.
"""
function anchor(response::T, anchor_var::S) where {T <: AbstractTerm, S <: AbstractTerm}
    return AnchorTerm{T, S}(response, anchor_var, nothing)
end
function anchor(response::T, anchor_var::S, h::Int) where {
        T <: AbstractTerm, S <: AbstractTerm,
    }
    return AnchorTerm{T, S}(response, anchor_var, h)
end

# termvars: Extract variables from anchor() for schema creation
function StatsModels.termvars(t::FunctionTerm{typeof(anchor)})
    length(t.args) >= 2 || return Symbol[]
    return unique(vcat(StatsModels.termvars(t.args[1]), StatsModels.termvars(t.args[2])))
end

# Struct for anchored response term
struct AnchorTerm{T <: AbstractTerm, S <: AbstractTerm} <: AbstractTerm
    response::T                  # The response term (can be nested like log(y))
    anchor::S                    # The anchor term (stays at time t)
    horizon::Union{Int, Nothing}  # nothing in lp() context, Int for standalone
end

StatsModels.terms(t::AnchorTerm) = (t.response, t.anchor)

function StatsModels.apply_schema(
        t::FunctionTerm{typeof(anchor)}, sch::StatsModels.Schema, ctx::Type
    )
    if length(t.args) == 2  # anchor(response, anchor_var) - horizon from context
        response, anchor_var = t.args
        horizon = nothing
    elseif length(t.args) == 3  # anchor(response, anchor_var, horizon)
        response, anchor_var, h_arg = t.args
        (h_arg isa ConstantTerm) ||
            throw(ArgumentError("anchor horizon must be a number (got $h_arg)"))
        horizon = h_arg.n
    else
        throw(ArgumentError("anchor() requires 2 or 3 arguments"))
    end

    response = StatsModels.apply_schema(response, sch, ctx)
    anchor_var = StatsModels.apply_schema(anchor_var, sch, ctx)
    return AnchorTerm{typeof(response), typeof(anchor_var)}(response, anchor_var, horizon)
end

function StatsModels.apply_schema(t::AnchorTerm, sch::StatsModels.Schema, ctx::Type)
    response = StatsModels.apply_schema(t.response, sch, ctx)
    anchor_var = StatsModels.apply_schema(t.anchor, sch, ctx)
    return AnchorTerm{typeof(response), typeof(anchor_var)}(response, anchor_var, t.horizon)
end

# modelcols: Apply anchored transformation (y_{t+h} - z_t)
function StatsModels.modelcols(at::AnchorTerm, d::Tables.ColumnTable)
    _check_horizon_provided(at.horizon, "anchor")
    response_cols = StatsModels.modelcols(at.response, d)
    anchor_cols = StatsModels.modelcols(at.anchor, d)
    response_cols = _extract_single_column(response_cols, "anchor() response")
    anchor_cols = _extract_single_column(anchor_cols, "anchor() anchor variable")
    return _create_anchored(response_cols, anchor_cols, at.horizon)
end

# width: Return number of columns (always 1 for anchor)
StatsModels.width(at::AnchorTerm) = 1

# show: Display the term
function Base.show(io::IO, at::AnchorTerm)
    return if at.horizon === nothing
        print(io, "anchor($(at.response), $(at.anchor))")
    else
        print(io, "anchor($(at.response), $(at.anchor), $(at.horizon))")
    end
end

# coefnames: Return coefficient name
function StatsModels.coefnames(at::AnchorTerm)
    response_names = StatsModels.coefnames(at.response)
    anchor_names = StatsModels.coefnames(at.anchor)
    if at.horizon === nothing
        return ["anchor(" * response_names[1] * ", " * anchor_names[1] * ")"]
    else
        return ["anchor(" * response_names[1] * ", " * anchor_names[1] * ", $(at.horizon))"]
    end
end

# termvars: Return variables used in anchor term
function StatsModels.termvars(at::AnchorTerm)
    return unique(vcat(StatsModels.termvars(at.response), StatsModels.termvars(at.anchor)))
end

# ============================================================================
# Pipe Operator (|) for Anchored Response Syntax
# Intercept | during schema application to create AnchorTerm
# ============================================================================

"""
    LPSchemaContext

Model context passed to `apply_schema` by [`lp`](@ref) and [`lpiv`](@ref).
The pipe syntax `y|z` is rewritten to an [`AnchorTerm`](@ref) only in this
context, so `|` keeps its StatsModels meaning in formulas used by other
packages. Subtyping `StatisticalModel` keeps every schema rule that dispatches
on `StatisticalModel` in effect.
"""
abstract type LPSchemaContext <: StatisticalModel end

"""
    apply_schema for FunctionTerm{typeof(|)}

Intercepts pipe operator in formulas to create AnchorTerm for anchored responses.
Enables syntax like:
- `@formula(leads(y)|z ~ x)` - equivalent to `@formula(anchor(y, z) ~ x)`
- `@formula(cumul(y)|z ~ x)` - cumulative response anchored to z

The pipe creates an AnchorTerm where:
- lhs is the response term (can be transformed: leads(y), cumul(y), log(y), etc.)
- rhs is the anchor term (stays fixed at time t)

# Examples
```julia
# Standard: y_{t+h}
lp(@formula(leads(y) ~ x), df; horizon=12)

# Anchored: y_{t+h} - z_t (pipe syntax)
lp(@formula(leads(y)|z ~ x), df; horizon=12)

```
"""
function StatsModels.apply_schema(
        t::FunctionTerm{typeof(|)}, sch::StatsModels.Schema, ctx::Type{<:LPSchemaContext}
    )
    # The pipe operator in formulas: lhs | rhs
    # Convert to AnchorTerm(lhs, rhs, nothing)
    if length(t.args) != 2
        throw(ArgumentError("Pipe operator | requires exactly 2 arguments (got $(length(t.args)))"))
    end

    lhs, rhs = t.args

    # Apply schema to both sides
    lhs_term = StatsModels.apply_schema(lhs, sch, ctx)
    rhs_term = StatsModels.apply_schema(rhs, sch, ctx)

    # Return AnchorTerm
    return AnchorTerm{typeof(lhs_term), typeof(rhs_term)}(lhs_term, rhs_term, nothing)
end

"""
    AbstractLocalProjection

Supertype of local projection results: [`LocalProjection`](@ref) (OLS, from
[`lp`](@ref)) and [`LocalProjectionIV`](@ref) (IV, from [`lpiv`](@ref)). Each holds
one fitted Regress.jl model per horizon `0:horizon`.

Per-horizon models are reached with [`horizonmodel`](@ref); [`coefnames`](@ref) and
[`nobs`](@ref) give the coefficient names and the sample size at each horizon.
"""
abstract type AbstractLocalProjection end

"""
    horizonmodel(lp::AbstractLocalProjection, h::Integer)

Return the fitted Regress.jl model for horizon `h`, where `0 ≤ h ≤ lp.horizon`.

The model supports the StatsAPI accessors (`coef`, `vcov`, `stderror`, `nobs`, ...).
Its coefficients are ordered as in `coefnames(lp)`. When the response is the shock,
the model at `h = 0` regresses the shock on itself; [`coefpath`](@ref) reports the
exact values (1 for the shock, 0 for other terms) at that horizon instead.

Throws an `ArgumentError` when `h` is outside `0:lp.horizon`.

# Example
```julia
r = lp(@formula(leads(y) ~ x), df; horizon = 8)
m = horizonmodel(r, 4)
coef(m)[findfirst(==("x"), coefnames(r))]
```
"""
function horizonmodel(lp::AbstractLocalProjection, h::Integer)
    0 <= h <= lp.horizon ||
        throw(ArgumentError("horizon $h is outside the estimated horizons 0:$(lp.horizon)"))
    return lp.models[h + 1]
end

"""
    coefnames(lp::AbstractLocalProjection) -> Vector{String}

Coefficient names of the per-horizon models. Every horizon has the same right-hand
side, so one vector serves all horizons.
"""
StatsModels.coefnames(lp::AbstractLocalProjection) = lp.coef_names

"""
    nobs(lp::AbstractLocalProjection) -> Vector{Int}

Number of observations used at each horizon; element `h + 1` belongs to horizon `h`.
With `sample = :varying` the count typically falls as `h` grows; with `sample = :fixed`
all entries are equal.
"""
StatsBase.nobs(lp::AbstractLocalProjection) = [nobs(m) for m in lp.models]

"""
    LocalProjection

Stack of horizon-specific OLS models produced by [`lp`](@ref).
"""
struct LocalProjection{M <: OLSMatrixEstimator} <: AbstractLocalProjection
    models::Vector{M}
    horizon::Int
    response::Symbol
    shock::Symbol
    base_formula::FormulaTerm
    coef_names::Vector{String}  # Coefficient names (constant across all horizons)
    tautological_h0::Bool       # true when response == shock (h=0 is trivial: coef=1, SE=0)
    sample::Symbol              # :varying (per-horizon rows) or :fixed (rows complete at all horizons)
    unit::Union{Symbol, Nothing}  # panel unit identifier column, if any
    time::Union{Symbol, Nothing}  # time column, if any
end

function Base.show(io::IO, lp::LocalProjection)
    return print(io, "LocalProjection(horizon=0:$(lp.horizon), response=$(lp.response), shock=$(lp.shock))")
end

function Base.show(io::IO, ::MIME"text/plain", lp::LocalProjection)
    println(io, "LocalProjection")
    println(io, "  Response:   $(lp.response)")
    println(io, "  Shock:      $(lp.shock)")
    println(io, "  Horizon:    0:$(lp.horizon)")
    println(io, "  Formula:    $(lp.base_formula)")
    println(io, "  Coef names: $(lp.coef_names)")
    println(io, "  Sample:     $(lp.sample)")
    lp.unit === nothing || println(io, "  Unit:       $(lp.unit)")
    lp.time === nothing || println(io, "  Time:       $(lp.time)")
    println(io, "  Vcov:       $(repr(_vcov_estimator(lp)))")
    return println(io, "  Obs (h=0:$(lp.horizon)): $(join(nobs(lp), ", "))")
end

"""
    lp + vcov(estimator)

Apply a covariance estimator to all models in a LocalProjection using the `+` operator.
Returns a new LocalProjection with updated variance-covariance specification for each model.

This is consistent with Regress.jl's `model + vcov(estimator)` pattern and allows chainable
operations like `(lp + vcov(A)) + vcov(B)`.

# Arguments
- `lp::LocalProjection`: The local projection result
- `v::VcovSpec{V}`: A variance-covariance specification from `vcov(estimator)`

# Returns
A new `LocalProjection` with the vcov estimator applied to each underlying model.

# Example
```julia
lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)

# Apply robust standard errors
lp_robust = lp_result + vcov(Bartlett{NeweyWest}())

# Coefficients unchanged, but models now have HAC vcov
@assert coefpath(lp_result) == coefpath(lp_robust)
```
"""
function Base.:+(lp::LocalProjection{M}, v::VcovSpec{V}) where {M <: OLSMatrixEstimator, V}
    # Apply vcov to each model using Regress.jl's + operator
    new_models = [m + v for m in lp.models]
    M_new = eltype(new_models)
    return LocalProjection{M_new}(
        convert(Vector{M_new}, new_models),
        lp.horizon, lp.response, lp.shock, lp.base_formula, lp.coef_names,
        lp.tautological_h0, lp.sample, lp.unit, lp.time
    )
end

"""
    _create_cumulative(y::AbstractVector, h::Int) -> Vector{Float64}

Helper function to compute cumulative sum from t to t+h for each observation.
Returns sum_{j=0}^{h} y_{t+j}.

At h=0, returns y itself (converted to Float64).
At h=1, returns y_t + y_{t+1}.
At h=2, returns y_t + y_{t+1} + y_{t+2}, etc.

NaN values are returned when:
- The sum cannot be computed (at the end of the series)
- Any component y_{t+j} is missing or NaN

# Returns
- `Vector{Float64}`: Type-stable Float64 vector with NaN at boundaries
"""
function _create_cumulative(y::AbstractVector, h::Int)::Vector{Float64}
    n = length(y)

    # Convert input to Float64, replacing missing with NaN
    # This handles both pure Float64 vectors and Union{Missing, Float64} vectors
    y_float = map(v -> ismissing(v) ? NaN : Float64(v), y)

    # Fast path for h=0
    h == 0 && return y_float

    result = Vector{Float64}(undef, n)
    @inbounds for t in 1:n
        if t + h > n
            result[t] = NaN
        else
            # Sum from t to t+h (inclusive, so h+1 values total)
            cumsum_val = 0.0
            all_valid = true
            for j in 0:h
                val = y_float[t + j]
                if isnan(val)
                    all_valid = false
                    break
                end
                cumsum_val += val
            end
            result[t] = all_valid ? cumsum_val : NaN
        end
    end

    return result
end

"""
    _create_anchored(y::AbstractVector, z::AbstractVector, h::Int) -> Vector{Float64}

Helper function to compute anchored response: y_{t+h} - z_t for each observation.

At h=0, returns y_t - z_t (both at time t).
At h=1, returns y_{t+1} - z_t (y shifted forward, z stays at t).
At h=2, returns y_{t+2} - z_t, etc.

The key difference from standard lead:
- Standard lead: y_{t+h} evolves freely
- Anchored: y_{t+h} - z_t measures deviation from anchor z_t

NaN values are returned when:
- Cannot compute lead (at the end of the series)
- Either y_{t+h} or z_t is missing or NaN

# Returns
- `Vector{Float64}`: Type-stable Float64 vector with NaN at boundaries
"""
function _create_anchored(y::AbstractVector, z::AbstractVector, h::Int)::Vector{Float64}
    n = length(y)
    length(z) == n || throw(ArgumentError("y and z must have same length"))
    return lead(y, h, default = NaN) .- z
end

"""

(term::AbstractTerm)

Recursively extract base variable names from a term, stripping away
function transformations like lag(), lead(), cumul().

Returns a vector of Symbol representing the raw variables referenced.

# Examples
```julia
_extract_base_variables(Term(:x))                    # [:x]
_extract_base_variables(FunctionTerm(lag, [:x, 4])) # [:x]  (strips lag)
```
"""
# Specialized methods for specific term types
function _extract_base_variables(
        t::Union{
            Term, StatsModels.ContinuousTerm, StatsModels.CategoricalTerm,
        }
    )
    return [t.sym]
end
_extract_base_variables(::ConstantTerm) = Symbol[]
_extract_base_variables(t::CumulTerm) = _extract_base_variables(t.term)
_extract_base_variables(t::LeadTerm) = _extract_base_variables(t.term)
function _extract_base_variables(t::AnchorTerm)
    return unique(vcat(_extract_base_variables(t.response), _extract_base_variables(t.anchor)))
end

function _extract_base_variables(ft::FunctionTerm)
    # For lag/lead/cumul/leads, extract the base variable (first argument)
    if ft.f in (lag, lead, cumul, leads)
        return _extract_base_variables(ft.args[1])
    elseif ft.f === anchor
        # For anchor, extract both response and anchor variables
        length(ft.args) >= 2 || return StatsModels.termvars(ft)
        return unique(
            vcat(
                _extract_base_variables(ft.args[1]),
                _extract_base_variables(ft.args[2])
            )
        )
    else
        # For other functions, use termvars
        return StatsModels.termvars(ft)
    end
end

function _extract_base_variables(terms::Tuple)
    # For RHS tuple of terms
    all_vars = Symbol[]
    for t in terms
        append!(all_vars, _extract_base_variables(t))
    end
    return unique(all_vars)
end

function _extract_base_variables(ft::FormulaTerm)
    return unique(
        vcat(
            _extract_base_variables(ft.lhs),
            _extract_base_variables(ft.rhs)
        )
    )
end

# InteractionTerm and other composite terms
function _extract_base_variables(t::StatsModels.InteractionTerm)
    all_vars = Symbol[]
    for component in t.terms
        append!(all_vars, _extract_base_variables(component))
    end
    return unique(all_vars)
end

"""
    _split_rhs(rhs) -> (exo, ivs)

Split an unapplied formula RHS into exogenous terms and `(endogenous ~ instruments)` terms.
"""
function _split_rhs(rhs)
    terms = rhs isa Tuple ? rhs : (rhs,)
    exo = Any[t for t in terms if !(t isa FormulaTerm)]
    ivs = FormulaTerm[t for t in terms if t isa FormulaTerm]
    return exo, ivs
end

"""
    _extract_all_vars_from_formula(formula::FormulaTerm) -> Vector{Symbol}

Extract all variable symbols from formula, including those nested in IV specs.
"""
function _extract_all_vars_from_formula(formula::FormulaTerm)
    all_vars = Symbol[]
    append!(all_vars, StatsModels.termvars(formula.lhs))
    exo, ivs = _split_rhs(formula.rhs)
    for term in exo
        append!(all_vars, StatsModels.termvars(term))
    end
    for term in ivs
        append!(all_vars, StatsModels.termvars(term.lhs))
        append!(all_vars, StatsModels.termvars(term.rhs))
    end
    return unique(all_vars)
end

"""
    _response_variable(lhs) -> Symbol

Name of the response variable, given the tuple returned by [`_unwrap_lhs`](@ref).
"""
function _response_variable(lhs)
    is_anchor, is_cumulative, is_leads, anchor_term, cumul_term, leads_term = lhs
    is_anchor && return _extract_single_response(anchor_term.response, "anchor() response term")
    is_cumulative && return _extract_single_response(cumul_term, "cumul() term")
    is_leads && return _extract_single_response(leads_term, "leads() term")
    throw(
        ArgumentError(
            "the response must use leads(), cumul(), anchor(), or the y|z syntax, " *
                "e.g. `leads(y) ~ x`"
        )
    )
end

"""
    _float_columns(cols) -> Matrix{Float64}

Convert model columns to a `Float64` matrix, mapping `missing` (from `lag`) to `NaN`.
"""
function _float_columns(cols)
    M = map(v -> ismissing(v) ? NaN : Float64(v), cols)
    return M isa AbstractMatrix ? Matrix{Float64}(M) : reshape(Vector{Float64}(M), :, 1)
end

"""
    _model_columns(t, data, groups) -> Matrix{Float64}

Model columns of the schema-applied term `t`, computed separately on each block of rows
in `groups` (the units of a panel) and stacked, so leads, lags and cumulative sums never
cross from one unit into the next.
"""
function _model_columns(t, data, groups)
    length(groups) == 1 && return _float_columns(StatsModels.modelcols(t, data))
    blocks = [
        _float_columns(StatsModels.modelcols(t, map(c -> view(c, g), data)))
            for g in groups
    ]
    return reduce(vcat, blocks)::Matrix{Float64}
end

"""
    _term_columns(terms, data, groups, nrows) -> (Matrix{Float64}, Vector{String})

Columns and coefficient names of schema-applied `terms`, without an intercept.
"""
_coefname_vector(t) = (n = StatsModels.coefnames(t); n isa AbstractVector ? n : [n])

function _term_columns(terms, data, groups, nrows::Integer)
    names = String[]
    cols = Matrix{Float64}[]
    for t in terms
        push!(cols, _model_columns(t, data, groups))
        append!(names, _coefname_vector(t))
    end
    isempty(cols) && return Matrix{Float64}(undef, nrows, 0), names
    return reduce(hcat, cols)::Matrix{Float64}, names
end

"""
    _missing_to_nan(tbl, vars) -> Tables.ColumnTable

Copy of `tbl` in which every numeric column among `vars` that admits `missing` is a
`Float64` column with `NaN` in place of `missing`. StatsModels cannot build model
columns that contain `missing`, while `NaN` rows are excluded per horizon by
[`_lp_estimate_horizons`](@ref).

Categorical (non-numeric) columns must not contain `missing`: their model columns
have no `NaN` to mark the row as unavailable.
"""
function _missing_to_nan(tbl, vars)
    cols = map(keys(tbl), values(tbl)) do name, col
        name in vars || return col
        if nonmissingtype(eltype(col)) <: Number
            return Missing <: eltype(col) ? map(v -> ismissing(v) ? NaN : Float64(v), col) : col
        end
        any(ismissing, col) && throw(
            ArgumentError(
                "categorical variable `$name` contains missing values; " *
                    "remove or recode them before estimation"
            )
        )
        return col
    end
    return NamedTuple{keys(tbl)}(cols)
end

_isunavailable(v) = ismissing(v) || (v isa AbstractFloat && isnan(v))

"""
    _panel_order(tbl, unit, time) -> (Tables.ColumnTable, Vector{UnitRange{Int}})

Sort the rows of `tbl` by `unit` and `time` and return the sorted table with the row
range of each unit. Without `time`, rows keep their order and form a single block.

Within a unit, consecutive rows must be consecutive periods of the time grid formed by
all values of `time` in the data. A repeated `(unit, time)` pair, or a unit that skips a
period present elsewhere in the data, is an error: leads and lags would otherwise shift
across the missing period.
"""
function _panel_order(tbl, unit, time)
    Base.require_one_based_indexing(values(tbl)...)
    nrows = isempty(tbl) ? 0 : length(first(tbl))
    if time === nothing
        unit === nothing || throw(
            ArgumentError(
                "`unit` requires `time`: rows are ordered by time within each unit"
            )
        )
        return tbl, [1:nrows]
    end
    for (kw, name) in ((:unit, unit), (:time, time))
        name === nothing && continue
        haskey(tbl, name) ||
            throw(ArgumentError("$kw column `$name` is not in the data"))
        any(_isunavailable, tbl[name]) && throw(
            ArgumentError("$kw column `$name` contains missing or NaN values")
        )
    end
    times = tbl[time]
    units = unit === nothing ? fill(nothing, nrows) : tbl[unit]
    perm = unit === nothing ? sortperm(times) : sortperm(collect(zip(units, times)))
    if perm != 1:nrows
        tbl = map(c -> c[perm], tbl)
        times, units = times[perm], units[perm]
    end

    grid = sort!(unique(times))
    period = Dict(t => i for (i, t) in pairs(grid))
    groups = UnitRange{Int}[]
    start = 1
    for i in 2:(nrows + 1)
        if i > nrows || !isequal(units[i], units[start])
            push!(groups, start:(i - 1))
            start = i
            continue
        end
        in_unit = unit === nothing ? "" : " in unit $(repr(units[i]))"
        step = period[times[i]] - period[times[i - 1]]
        step == 0 && throw(
            ArgumentError("time $(repr(times[i])) appears in more than one row$in_unit")
        )
        step == 1 || throw(
            ArgumentError(
                "no row at time $(repr(grid[period[times[i - 1]] + 1]))$in_unit, " *
                    "between $(repr(times[i - 1])) and $(repr(times[i])); add a row " *
                    "with missing values for that period so leads and lags do not skip it"
            )
        )
    end
    return tbl, groups
end

"""
    _select_shock(shock, coef_names, endo_names) -> Symbol

Validate a user-supplied `shock`, or pick the default: the first endogenous regressor
if there is one, otherwise the first coefficient other than the intercept.
"""
function _select_shock(shock, coef_names, endo_names)
    if shock === nothing
        candidates = isempty(endo_names) ? filter(!=("(Intercept)"), coef_names) : endo_names
        isempty(candidates) && throw(
            ArgumentError(
                "cannot select a default shock: the model has no regressor besides the " *
                    "intercept; add a regressor or pass `shock`"
            )
        )
        return Symbol(first(candidates))
    end
    Symbol(shock) in Symbol.(coef_names) || throw(
        ArgumentError(
            "shock `$shock` is not a coefficient of the model; " *
                "available coefficients: $(join(coef_names, ", "))"
        )
    )
    return Symbol(shock)
end

"""
    _lp_design(formula, data, horizon, shock, instrumented::Bool, unit, time)

Parse `formula` and build the matrices shared by every horizon of [`lp`](@ref) and
[`lpiv`](@ref). `instrumented` states whether `(endogenous ~ instruments)` terms are
required (`lpiv`) or forbidden (`lp`). `unit` and `time` name the panel columns, or are
`nothing`; see [`_panel_order`](@ref).

Returns a `NamedTuple` with the regressor matrix `X = [exogenous endogenous]`, the
instrument matrix `Z = [exogenous instruments]` (equal to `X` without instruments),
coefficient, endogenous and instrument names, the response name, the unwrapped LHS,
the data as a `Tables.ColumnTable` in time order, the row range of each unit, and the
validated shock.

Rows are never dropped here: leads must be computed on the full table. `missing` in
numeric columns becomes `NaN` and is excluded per horizon by
[`_lp_estimate_horizons`](@ref).
"""
function _lp_design(
        formula::FormulaTerm, data, horizon::Integer, shock, instrumented::Bool, unit, time
    )
    horizon < 0 && throw(ArgumentError("horizon must be non-negative (got $horizon)"))
    Tables.istable(data) || throw(
        ArgumentError(
            "data must be a Tables.jl table, e.g. a DataFrame or a NamedTuple of vectors " *
                "(got $(typeof(data)))"
        )
    )
    all_vars = _extract_all_vars_from_formula(formula)
    tbl, groups = _panel_order(Tables.columntable(data), unit, time)
    tbl = _missing_to_nan(tbl, all_vars)

    exo_raw, iv_raw = _split_rhs(formula.rhs)
    if instrumented
        isempty(iv_raw) && throw(
            ArgumentError(
                "lpiv requires an `(endogenous ~ instruments)` term, e.g. `leads(y) ~ (x ~ z)`"
            )
        )
    else
        isempty(iv_raw) || throw(
            ArgumentError(
                "lp does not accept `(endogenous ~ instruments)` terms; " *
                    "use `lpiv(TSLS(), formula, data; horizon)` for IV local projections"
            )
        )
    end

    sch = StatsModels.schema(formula, tbl)

    lhs = _unwrap_lhs(StatsModels.apply_schema(formula.lhs, sch, LPSchemaContext))
    response = _response_variable(lhs)
    nrows = length(tbl[response])

    isempty(StatsModels.termvars(formula.rhs)) &&
        throw(ArgumentError("formula must contain at least one regressor"))

    # Exogenous regressors, including the intercept unless the formula removes it
    exo_rhs = isempty(exo_raw) ? ConstantTerm(1) : Tuple(exo_raw)
    exo_f = StatsModels.apply_schema(FormulaTerm(Term(response), exo_rhs), sch, StatisticalModel)
    X_exo = _model_columns(exo_f.rhs, tbl, groups)
    exo_names = Vector{String}(_coefname_vector(exo_f.rhs))

    endo_terms = AbstractTerm[]
    instr_terms = AbstractTerm[]
    for t in iv_raw
        for e in (t.lhs isa Tuple ? t.lhs : (t.lhs,))
            push!(endo_terms, StatsModels.apply_schema(e, sch, LPSchemaContext))
        end
        for z in (t.rhs isa Tuple ? t.rhs : (t.rhs,))
            push!(instr_terms, StatsModels.apply_schema(z, sch, LPSchemaContext))
        end
    end
    X_endo, endo_names = _term_columns(endo_terms, tbl, groups, nrows)
    Z_instr, instr_names = _term_columns(instr_terms, tbl, groups, nrows)
    length(instr_names) >= length(endo_names) || throw(
        ArgumentError(
            "Not enough instruments: $(length(instr_names)) < $(length(endo_names)) " *
                "(order condition violated)"
        )
    )

    coef_names = vcat(exo_names, endo_names)
    return (;
        X = hcat(X_exo, X_endo)::Matrix{Float64},
        Z = hcat(X_exo, Z_instr)::Matrix{Float64},
        coef_names, endo_names, instr_names, response, lhs,
        data = tbl, groups,
        shock = _select_shock(shock, coef_names, endo_names),
    )
end

_fit_horizon(::Nothing, X, Z, y, n_endogenous) = ols(X, y; has_intercept = false)
function _fit_horizon(estimator, X, Z, y, n_endogenous)
    return iv(
        estimator, collect(Z), collect(X), collect(y);
        has_intercept = false, n_endogenous
    )
end

"""
    _lp_estimate_horizons(estimator, X, Z, n_endogenous, lhs, data, groups, horizon, sample)

Fit one model per horizon `0:horizon`: OLS when `estimator === nothing`, otherwise
`Regress.iv(estimator, ...)`. With `sample = :varying`, each horizon uses the rows where
`X`, `Z` and the transformed response at that horizon are all non-`NaN`; with
`sample = :fixed`, every horizon uses the rows where they are non-`NaN` at all horizons.
The response is computed within each block of rows in `groups`.

`X` and `Z` are typed `Matrix{Float64}` so the loop infers concretely despite the
abstract return types of StatsModels.
"""
function _lp_estimate_horizons(
        estimator, X::Matrix{Float64}, Z::Matrix{Float64}, n_endogenous::Int,
        lhs, data, groups, horizon::Integer, sample::Symbol
    )
    sample in (:varying, :fixed) || throw(
        ArgumentError("sample must be :varying or :fixed (got $(repr(sample)))")
    )
    complete_xz = vec(all(!isnan, X, dims = 2)) .& vec(all(!isnan, Z, dims = 2))
    # Vector{Float64}: modelcols may return a ShiftedArray or contain `missing`
    ys = [
        _model_columns(_build_lhs_for_horizon(h, lhs...), data, groups)[:, 1]
            for h in 0:horizon
    ]
    rows = [complete_xz .& .!isnan.(y) for y in ys]
    if sample === :fixed
        common = reduce(.&, rows)
        any(common) || throw(
            ArgumentError("No observations are complete at every horizon 0:$horizon")
        )
        fill!(rows, common)
    end

    function _estimate_horizon(h)
        r, y_h = rows[h + 1], ys[h + 1]
        any(r) ||
            throw(ArgumentError("No complete observations available for horizon $h"))
        return _fit_horizon(
            estimator, view(X, r, :), view(Z, r, :), view(y_h, r), n_endogenous
        )
    end

    first_model = _estimate_horizon(0)
    models = Vector{typeof(first_model)}(undef, horizon + 1)
    models[1] = first_model
    for h in 1:horizon
        models[h + 1] = _estimate_horizon(h)
    end
    return models
end

"""
    lp(formula, data; horizon, shock=nothing, sample=:varying, unit=nothing, time=nothing)
        -> LocalProjection

Estimate OLS local projections from horizon 0 to `horizon`, one regression per horizon.

`data` is any Tables.jl table. Numeric columns enter as continuous variables; other
columns (strings, `CategoricalArray`s) enter as categorical variables through
StatsModels contrasts.

Without `time`, row order is time order. With `time = :t`, rows are sorted by the
column `t` first, and a time value that appears twice is an error. For a panel, `unit`
names the column identifying the cross-sectional unit (it requires `time`): rows are
sorted by unit and time, and leads, lags, cumulative sums and anchors are computed
within each unit, so no value is shifted from one unit into another. The time values
found anywhere in the data form the time grid; a unit that skips a period of that grid
between its first and last observation is an error (add a row with missing values for
the period instead). Units may start and end at different times. A single series has no
other units to reveal a skipped period, so its rows are taken as consecutive. Unit fixed effects
enter as a categorical control, e.g. `leads(y) ~ x + id` with `id` a string or
`CategoricalArray` column.

The response must be written with `leads(y)`, `cumul(y)`, `anchor(y, z)` or the pipe
syntax `y|z`. `shock` names the coefficient whose path is reported by
[`coefpath`](@ref) and [`summarize`](@ref); by default it is the first regressor other
than the intercept. A `shock` that is not a coefficient of the model is an error.

Rows are kept in place, and `missing` or `NaN` values exclude a row only from the
horizons that need them. With `sample = :varying` (the default), horizon `h` uses the
rows where the regressors and the response at `t + h` are available, so the sample
shrinks as `h` grows. With `sample = :fixed`, every horizon uses the rows that are
available at all horizons `0:horizon`. The per-horizon sample sizes are shown when the
result is displayed.

For instrumental variables, use [`lpiv`](@ref).

# Example
```julia
result = lp(@formula(leads(y) ~ x + lags(w, 4)), df; horizon = 12)
coefpath(result)            # path of `x`
```
"""
function lp(
        formula::FormulaTerm, data;
        horizon::Integer, shock::Union{Symbol, Nothing} = nothing, sample::Symbol = :varying,
        unit::Union{Symbol, Nothing} = nothing, time::Union{Symbol, Nothing} = nothing
    )
    d = _lp_design(formula, data, horizon, shock, false, unit, time)
    models = _lp_estimate_horizons(
        nothing, d.X, d.Z, 0, d.lhs, d.data, d.groups, horizon, sample
    )
    return LocalProjection(
        models, horizon, d.response, d.shock, formula, d.coef_names,
        d.response === d.shock, sample, unit, time
    )
end

"""
    IRFSummary

Impulse response of `response` to `term` at horizons `0:H`: coefficients, standard
errors and `level` confidence bounds, all multiplied by `scale`. Built by
[`summarize`](@ref).

An `IRFSummary` is a Tables.jl table with one row per horizon and columns `response`,
`term`, `horizon`, `coef`, `se`, `lower`, `upper`, `level`, `scale`, so
`DataFrame(summarize(lp))` works. Its text display marks in bold the horizons whose
confidence interval excludes zero, when the output supports color.
"""
struct IRFSummary
    response::Symbol
    term::Symbol
    level::Float64
    scale::Float64
    horizon::Vector{Int}
    coef::Vector{Float64}
    se::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

Tables.istable(::Type{IRFSummary}) = true
Tables.columnaccess(::Type{IRFSummary}) = true

function Tables.columns(s::IRFSummary)
    n = length(s.horizon)
    return (
        response = fill(s.response, n), term = fill(s.term, n), horizon = s.horizon,
        coef = s.coef, se = s.se, lower = s.lower, upper = s.upper,
        level = fill(s.level, n), scale = fill(s.scale, n),
    )
end

function Base.show(io::IO, s::IRFSummary)
    return print(
        io, "IRFSummary(response=$(s.response), term=$(s.term), level=$(s.level), scale=$(s.scale))"
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::IRFSummary)
    significant = (s.lower .> 0) .| (s.upper .< 0)
    data = Any[s.horizon s.coef s.se s.lower s.upper]
    level_pct = round(s.level * 100; digits = 2)
    level_str = isinteger(level_pct) ? string(Int(level_pct)) : string(level_pct)
    labels = ["Horizon", "Coef", "Std.Err.", "Lower $level_str%", "Upper $level_str%"]

    # Bold coef, lower and upper where the interval excludes zero
    hl = TextHighlighter(
        (d, i, j) -> j in (2, 4, 5) && significant[i],
        crayon"bold"
    )

    title = "Impulse response of $(s.response) to $(s.term)"
    if s.scale != 1.0
        title *= " (scale=$(s.scale))"
    end

    return pretty_table(
        io, data;
        column_labels = labels,
        title = title,
        highlighters = [hl],
        formatters = [(v, i, j) -> v isa AbstractFloat ? round(v; digits = 4) : v],
        alignment = [:r, :r, :r, :r, :r],
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded)
    )
end

# ============================================================================
# Instrumental Variables Local Projections
# ============================================================================

"""
    LocalProjectionIV

Stack of horizon-specific IV models produced by [`lpiv`](@ref).
"""
struct LocalProjectionIV{M <: IVMatrixEstimator} <: AbstractLocalProjection
    models::Vector{M}
    horizon::Int
    response::Symbol
    shock::Symbol
    base_formula::FormulaTerm
    coef_names::Vector{String}  # Coefficient names (constant across all horizons)
    endogenous_names::Vector{String}
    instrument_names::Vector{String}
    tautological_h0::Bool       # true when response == shock (h=0 is trivial: coef=1, SE=0)
    sample::Symbol              # :varying (per-horizon rows) or :fixed (rows complete at all horizons)
    unit::Union{Symbol, Nothing}  # panel unit identifier column, if any
    time::Union{Symbol, Nothing}  # time column, if any
end

function Base.show(io::IO, lpiv::LocalProjectionIV)
    return print(
        io,
        "LocalProjectionIV(horizon=0:$(lpiv.horizon), response=$(lpiv.response), shock=$(lpiv.shock))"
    )
end

function Base.show(io::IO, ::MIME"text/plain", lpiv::LocalProjectionIV)
    println(io, "LocalProjectionIV")
    println(io, "  Response:     $(lpiv.response)")
    println(io, "  Shock:        $(lpiv.shock)")
    println(io, "  Horizon:      0:$(lpiv.horizon)")
    println(io, "  Formula:      $(lpiv.base_formula)")
    println(io, "  Endogenous:   $(lpiv.endogenous_names)")
    println(io, "  Instruments:  $(lpiv.instrument_names)")
    println(io, "  Coef names:   $(lpiv.coef_names)")
    println(io, "  Sample:       $(lpiv.sample)")
    lpiv.unit === nothing || println(io, "  Unit:         $(lpiv.unit)")
    lpiv.time === nothing || println(io, "  Time:         $(lpiv.time)")
    println(io, "  Vcov:         $(repr(_vcov_estimator(lpiv)))")
    return println(io, "  Obs (h=0:$(lpiv.horizon)): $(join(nobs(lpiv), ", "))")
end

"""
    lpiv + vcov(estimator)

Apply a covariance estimator to all models in a LocalProjectionIV using the `+` operator.
Returns a new LocalProjectionIV with updated variance-covariance specification for each model.

# Example
```julia
result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon=10)
result_hac = result + vcov(Bartlett{NeweyWest}())
```
"""
function Base.:+(lpiv::LocalProjectionIV{M}, v::VcovSpec{V}) where {
        M <: IVMatrixEstimator, V,
    }
    new_models = [m + v for m in lpiv.models]
    M_new = eltype(new_models)
    return LocalProjectionIV{M_new}(
        convert(Vector{M_new}, new_models),
        lpiv.horizon, lpiv.response, lpiv.shock, lpiv.base_formula,
        lpiv.coef_names, lpiv.endogenous_names, lpiv.instrument_names,
        lpiv.tautological_h0, lpiv.sample, lpiv.unit, lpiv.time
    )
end

"""
    lpiv(estimator, formula, data; horizon, shock=nothing, sample=:varying,
         unit=nothing, time=nothing) -> LocalProjectionIV

Estimate instrumental-variables local projections from horizon 0 to `horizon` with a
Regress.jl IV `estimator`, e.g. `TSLS()`.

# Formula Syntax
```julia
lpiv(TSLS(), @formula(leads(y) ~ (x ~ z1) + lags(r, 5) + w), df; horizon=10)
```

Where:
- `leads(y)` - response with horizon transformation (also supports `cumul(y)`, `anchor(y, baseline)`)
- `(x ~ z1)` - x is endogenous, instrumented by z1
- `lags(r, 5)` and `w` - exogenous controls

# Arguments
- `estimator`: a Regress.jl IV estimator with a matrix-based `Regress.iv` method (`TSLS()`)
- `formula::FormulaTerm`: A formula with IV specification using `(endo ~ instruments)` syntax
- `data`: a Tables.jl table, handled as in [`lp`](@ref)
- `horizon::Integer`: Maximum forecast horizon (must be non-negative)
- `shock::Union{Symbol, Nothing}`: coefficient to track; defaults to the first endogenous
  variable. A `shock` that is not a coefficient of the model is an error.
- `sample::Symbol`: `:varying` (per-horizon samples) or `:fixed` (rows available at
  every horizon), as in [`lp`](@ref)
- `unit`, `time`: panel unit and time columns, as in [`lp`](@ref)

# Returns
`LocalProjectionIV` struct containing IV models for each horizon.

# Example
```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices

# Generate data with endogeneity
n = 200
z = randn(n)
u = randn(n)
x = 0.5*z + 0.5*u      # Endogenous (correlated with u)
y = 1.0 .+ 2.0*x .+ u  # True effect = 2.0

df = DataFrame(y=y, x=x, z=z)

# Fit IV local projection
result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z)), df; horizon=5)

# Extract IRF (should be ~2.0, OLS would be biased)
irf = coefpath(result; term=:x)

# Apply HAC standard errors
result_hac = result + vcov(Bartlett{NeweyWest}())

# First-stage OLS regression (one per endogenous variable)
fs = first_stage(result, 0)
println("First-stage R²: ", r2(fs[1]))
println("First-stage coefs: ", coef(fs[1]))
```
"""
function lpiv(
        estimator::AbstractIVEstimator, formula::FormulaTerm, data;
        horizon::Integer, shock::Union{Symbol, Nothing} = nothing, sample::Symbol = :varying,
        unit::Union{Symbol, Nothing} = nothing, time::Union{Symbol, Nothing} = nothing
    )
    hasmethod(iv, Tuple{typeof(estimator), Matrix{Float64}, Matrix{Float64}, Vector{Float64}}) ||
        throw(
        ArgumentError(
            "Regress.jl has no matrix-based `iv` method for $(nameof(typeof(estimator)))"
        )
    )
    d = _lp_design(formula, data, horizon, shock, true, unit, time)
    models = _lp_estimate_horizons(
        estimator, d.X, d.Z, length(d.endo_names), d.lhs, d.data, d.groups, horizon, sample
    )
    return LocalProjectionIV(
        models, horizon, d.response, d.shock, formula, d.coef_names,
        d.endo_names, d.instr_names, d.response === d.shock, sample, unit, time
    )
end

"""
    first_stage(lpiv::LocalProjectionIV)

Return first-stage regressions and diagnostics for all horizons.

Returns a vector of `FirstStageIV`, one per horizon (0 to `lpiv.horizon`).
Each contains full OLS models plus non-robust and robust first-stage F-statistics.
"""
function Regress.first_stage(lpiv::LocalProjectionIV)
    return [first_stage(lpiv, h) for h in 0:lpiv.horizon]
end

"""
    first_stage(lpiv::LocalProjectionIV, h::Int) -> FirstStageIV

Return first-stage regressions and diagnostics for horizon `h` (0-indexed).

# Example
```julia
result = lpiv(TSLS(), @formula(leads(y) ~ (x ~ z1 + z2)), df; horizon=10)
fs = first_stage(result, 0)
coef(fs.models[1])    # first-stage coefficients
fs.F_nonrobust        # non-robust F
fs.F_robust           # robust F using the model's current vcov
```
"""
function Regress.first_stage(lpiv::LocalProjectionIV, h::Int)
    return Regress.first_stage(
        horizonmodel(lpiv, h);
        endogenous_names = lpiv.endogenous_names,
        instrument_names = lpiv.instrument_names
    )
end

# ============================================================================
# Weak IV Test for LocalProjectionIV
# ============================================================================

"""
    weakivtest(lpiv::LocalProjectionIV, h::Int; kwargs...) -> WeakIVTestResult

Montiel-Olea-Pflueger robust weak instrument test for horizon `h` (0-indexed).

When the response is the endogenous shock, horizon 0 is tautological (the coefficient
is exactly 1) and there is no IV estimate to test; `weakivtest(lpiv, 0)` then throws an
`ArgumentError`. `first_stage(lpiv, 0)` remains available.

# Keyword Arguments
- `level::Real=0.05`: Confidence level alpha
- `eps::Real=0.001`: Convergence tolerance for bias optimization
- `benchmark::Symbol=:nagar`: Bias benchmark (`:nagar` or `:ols`)

# Returns
A `WeakIVTestResult` containing effective F, robust F, critical values, etc.

See also: [`lpiv`](@ref), [`first_stage`](@ref)
"""
function Regress.weakivtest(lpiv::LocalProjectionIV, h::Int; kwargs...)
    m = horizonmodel(lpiv, h)
    if h == 0 && lpiv.tautological_h0
        throw(
            ArgumentError(
                "weakivtest is undefined at horizon 0 when the response equals the shock " *
                    "($(lpiv.shock)): the horizon-0 coefficient is 1 by construction. " *
                    "Use first_stage(lpiv, 0) for first-stage diagnostics."
            )
        )
    end
    return Regress.weakivtest(m; kwargs...)
end

"""
    weakivtest(lpiv::LocalProjectionIV; kwargs...) -> Vector

Montiel-Olea-Pflueger robust weak instrument test for all horizons.

Returns a vector with one `WeakIVTestResult` per horizon `0:lpiv.horizon`. When the
response is the endogenous shock, the entry for horizon 0 is `nothing` (see
`weakivtest(lpiv, h)`), so the vector stays aligned with the horizons.
"""
function Regress.weakivtest(lpiv::LocalProjectionIV; kwargs...)
    return [
        h == 0 && lpiv.tautological_h0 ? nothing : weakivtest(lpiv, h; kwargs...)
            for h in 0:lpiv.horizon
    ]
end

# ============================================================================
# Shared methods for LocalProjection and LocalProjectionIV
# ============================================================================

function _term_index(lp::AbstractLocalProjection, term::Symbol)
    idx = findfirst(==(String(term)), lp.coef_names)
    idx === nothing && throw(ArgumentError("term $term not present in model"))
    return idx
end

"""
    coefpath(lp; term = lp.shock) -> Vector{Float64}

Coefficient of `term` across horizons; element `h + 1` belongs to horizon `h`.
When the response is the shock, the value at `h = 0` is exactly 1 for the shock and 0
for every other term.
"""
function coefpath(lp::AbstractLocalProjection; term::Symbol = lp.shock)
    idx = _term_index(lp, term)
    coefficients = [coef(horizonmodel(lp, h))[idx] for h in 0:lp.horizon]
    if lp.tautological_h0
        coefficients[1] = term === lp.shock ? 1.0 : 0.0
    end
    return coefficients
end

"""
    stderror(lp::AbstractLocalProjection; term = lp.shock) -> Vector{Float64}

Standard errors of `term` across horizons; element `h + 1` belongs to horizon `h`.

Each horizon uses the covariance estimator stored in its model: HC1 after [`lp`](@ref)
or [`lpiv`](@ref), or the one applied with `lp + vcov(estimator)`. When the response is
the shock, the standard error at `h = 0` is exactly 0 (see [`coefpath`](@ref)).

# Example
```julia
r = lp(@formula(leads(y) ~ x), df; horizon = 8)
stderror(r)                                  # HC1
stderror(r + vcov(Bartlett{NeweyWest}()))    # HAC
```
"""
function StatsBase.stderror(lp::AbstractLocalProjection; term::Symbol = lp.shock)
    idx = _term_index(lp, term)
    se = [stderror(horizonmodel(lp, h))[idx] for h in 0:lp.horizon]
    if lp.tautological_h0
        se[1] = 0.0
    end
    return se
end

# Every horizon model carries the same estimator: `lp` fits all horizons with the
# default and `lp + vcov(estimator)` replaces it on all of them.
_vcov_estimator(lp::AbstractLocalProjection) = horizonmodel(lp, 0).vcov_estimator

"""
    summarize(lp; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary
    summarize(lp, estimator; term=lp.shock, level=0.95, scale=1.0) -> IRFSummary

Table of the impulse response of `term` with standard errors and `level` confidence
intervals, all multiplied by `scale`; `level` must lie strictly between 0 and 1. The
numbers are those of [`as_irf_result`](@ref) with `coverage = [level]`. A negative
`scale` flips the sign of the response, and the bounds swap so that `lower <= upper`.

Standard errors come from the covariance estimator stored in `lp` (see
[`stderror`](@ref)). Passing a CovarianceMatrices.jl `estimator` is the same as
`summarize(lp + vcov(estimator); ...)`.
"""
function summarize(
        lp::AbstractLocalProjection;
        term::Symbol = lp.shock, level::Real = 0.95, scale::Real = 1.0
    )
    irf = _irf_result(lp, term, (level,); diagnostics = false)
    scale_f = Float64(scale)
    lower = scale_f .* _horizon_vector(only(lowerbounds(irf)))
    upper = scale_f .* _horizon_vector(only(upperbounds(irf)))
    # A negative scale reverses the order of the bounds
    if scale_f < 0
        lower, upper = upper, lower
    end
    return IRFSummary(
        lp.response, term, only(coverages(irf)), scale_f, collect(0:lp.horizon),
        scale_f .* _horizon_vector(point_estimate(irf)),
        abs(scale_f) .* _horizon_vector(irf.stderr),
        lower, upper
    )
end

function summarize(
        lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        kwargs...
    )
    return summarize(lp + vcov(estimator); kwargs...)
end

# ============================================================================
# Plotting
# ============================================================================

# Every plotting path draws the IRF object built by `_irf_result`, using
# MacroEconometricTools' Makie methods and Plots recipe for that object.

"""
    irfplot(lp; term = lp.shock, levels = [0.95], kwargs...) -> Makie.Figure
    irfplot(lp, estimator; term = lp.shock, levels = [0.95], kwargs...)

Plot the impulse response of `lp.response` to `term`, with a confidence band for each
of `levels`. Requires a Makie backend (e.g. `CairoMakie`).

Bands use the estimator stored in `lp`; passing `estimator` plots
`lp + vcov(estimator)`. They are the bands of [`as_irf_result`](@ref) with
`coverage = levels`. The figure is drawn by MacroEconometricTools' `irfplot` for that
object, which receives the remaining keywords (`bandcolor`, `bandalpha`, `linecolor`,
`linewidth`, `drawzero`, `xtickstep`, `title`, `size`, ...).

With Plots.jl, `plot(lp; term, levels)` and `plot(lp, estimator; term, levels)` draw
the same object with MacroEconometricTools' Plots recipe.
"""
function MacroEconometricTools.irfplot(
        lp::AbstractLocalProjection; term::Symbol = lp.shock, levels = [0.95], kwargs...
    )
    return irfplot(_irf_result(lp, term, levels; diagnostics = false); kwargs...)
end

function MacroEconometricTools.irfplot(
        lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        kwargs...
    )
    return irfplot(lp + vcov(estimator); kwargs...)
end

"""
    irfplot!(ax, lp; term = lp.shock, levels = [0.95], kwargs...) -> ax
    irfplot!(ax, lp, estimator; term = lp.shock, levels = [0.95], kwargs...)

Draw the impulse response of `lp.response` to `term` onto the Makie `Axis` `ax`.
Bands and keywords are as in [`irfplot`](@ref); MacroEconometricTools' `irfplot!`
draws the panel.
"""
function MacroEconometricTools.irfplot!(
        ax, lp::AbstractLocalProjection; term::Symbol = lp.shock, levels = [0.95], kwargs...
    )
    irf = _irf_result(lp, term, levels; diagnostics = false)
    return irfplot!(ax, irf; var = lp.response, shock = term, kwargs...)
end

function MacroEconometricTools.irfplot!(
        ax, lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        kwargs...
    )
    return irfplot!(ax, lp + vcov(estimator); kwargs...)
end

@recipe function f(lp::AbstractLocalProjection; term = lp.shock, levels = [0.95])
    return _irf_result(lp, term, levels; diagnostics = false)
end

@recipe function f(
        lp::AbstractLocalProjection,
        estimator::CovarianceMatrices.AbstractAsymptoticVarianceEstimator;
        term = lp.shock, levels = [0.95]
    )
    return _irf_result(lp + vcov(estimator), term, levels; diagnostics = false)
end

# ============================================================================
# MacroEconometricTools Integration - LocalProjectionIRFResult Conversion
# ============================================================================

"""
    as_irf_result(lp; term=lp.shock, vcov_estimator=nothing, coverage=[0.68, 0.90, 0.95])

Convert a `LocalProjection` or `LocalProjectionIV` result to MET's `LocalProjectionIRFResult` type.

This enables using MET's unified plotting infrastructure with AxisArray support.
Bands are `coef ± z * se`, with `z` the standard normal quantile for each coverage level
and `se` from the covariance estimator. For IV results, first-stage diagnostics are
included in metadata.

# Keyword Arguments
- `term::Symbol=lp.shock`: Which coefficient to extract (defaults to shock variable)
- `vcov_estimator=nothing`: Covariance estimator for standard errors and bands (e.g.,
  `HR1()`); `nothing` uses the estimator stored in `lp` (see [`stderror`](@ref))
- `coverage=[0.68, 0.90, 0.95]`: confidence levels for the bands, each strictly between
  0 and 1

# Returns
- `LocalProjectionIRFResult`: MET-compatible IRF result with AxisArray data

# Example
```julia
using LocalProjections, MacroEconometricTools, CovarianceMatrices

lp_result = lp(@formula(leads(y) ~ x), df; horizon=12)
irf = as_irf_result(lp_result; vcov_estimator=HR1())

# AxisArray indexing
irf.data[response=:y, shock=:x, horizon=0:6]

# Use with MET's plotting
using Plots
plot(irf)  # Uses MET's recipe
```
"""
function as_irf_result(
        lp::AbstractLocalProjection;
        term::Symbol = lp.shock,
        vcov_estimator = nothing,
        coverage = [0.68, 0.9, 0.95]
    )
    if vcov_estimator !== nothing
        return as_irf_result(lp + vcov(vcov_estimator); term, coverage)
    end
    return _irf_result(lp, term, coverage; diagnostics = true)
end

function _check_levels(levels)
    isempty(levels) && throw(ArgumentError("at least one confidence level is required"))
    for c in levels
        0 < c < 1 || throw(
            ArgumentError("confidence levels must lie strictly between 0 and 1; got $c")
        )
    end
    return nothing
end

# The response × shock × horizon cube of a single (response, term) pair, as a vector
# over horizons 0:H.
_horizon_vector(a::AxisArray) = vec(Array(a))

# Coefficient path, standard errors and normal confidence bands for `term` at each level
# in `coverage`, using the covariance estimator stored in `lp`. Every confidence bound
# the package reports (`summarize`, `as_irf_result`) is computed here. With
# `diagnostics = true`, IV results also carry the first-stage regressions per horizon.
function _irf_result(lp::AbstractLocalProjection, term::Symbol, coverage; diagnostics::Bool)
    levels = collect(Float64, coverage)
    _check_levels(levels)
    beta = coefpath(lp; term)
    se = stderror(lp; term)
    cube(v) = AxisArray(
        reshape(v, 1, 1, length(v)),
        Axis{:response}([lp.response]),
        Axis{:shock}([term]),
        Axis{:horizon}(0:lp.horizon)
    )
    z = [quantile(Normal(), 0.5 + c / 2) for c in levels]
    lower = [cube(beta .- zc .* se) for zc in z]
    upper = [cube(beta .+ zc .* se) for zc in z]

    meta = (horizon = lp.horizon, formula = lp.base_formula, term = term)
    if diagnostics && lp isa LocalProjectionIV
        meta = (meta..., is_iv = true, first_stage = [first_stage(lp, h) for h in 0:lp.horizon])
    end
    return LocalProjectionIRFResult(cube(beta), cube(se), lower, upper, levels, meta)
end

end # module LocalProjections
