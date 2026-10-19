module LocalProjectionsMakieExt

using Makie
using LocalProjections
using LocalProjections: AbstractLocalProjection, coefpath, stderror, vcov
using CovarianceMatrices
using Distributions: Normal, quantile

import LocalProjections: irfplot, irfplot!, irfplot_axis

# ============================================================================
# IRFPlotMakie Recipe
# ============================================================================

Makie.@recipe(IRFPlotMakie, lp) do scene
    Makie.Theme(;
        term = Makie.automatic,
        levels = [0.95],
        drawzero = true,
        zerolinecolor = :gray70,
        bandcolor = :blue,
        bandalpha = 0.25,
        linecolor = :black,
        linewidth = 2.0,
        xtickstep = 4
    )
end

function Makie.plot!(plot::IRFPlotMakie)
    lp = plot[1][]

    term_val = plot.term[]
    term = term_val === Makie.automatic ? lp.shock : term_val
    levels = plot.levels[]
    drawzero = plot.drawzero[]
    zerolinecolor = plot.zerolinecolor[]
    bandcolor = plot.bandcolor[]
    bandalpha = plot.bandalpha[]
    linecolor = plot.linecolor[]
    linewidth = plot.linewidth[]
    xtickstep = plot.xtickstep[]

    beta = coefpath(lp; term = term)
    se = stderror(lp; term = term)
    horizons = collect(0:lp.horizon)

    sorted_levels = sort(Float64.(levels); rev = true)
    for level in sorted_levels
        (level <= 0 || level >= 1) &&
            throw(ArgumentError("levels must be in (0, 1), got $level"))
    end

    # Draw confidence bands (widest first, fading alpha)
    for (idx, level) in enumerate(sorted_levels)
        z = quantile(Normal(), 0.5 + level / 2)
        lower = beta .- z .* se
        upper = beta .+ z .* se
        alpha = clamp(bandalpha * 0.8^(idx - 1), 0.0f0, 1.0f0)
        Makie.band!(plot, horizons, lower, upper; color = (bandcolor, alpha))
    end

    # Point estimate line
    Makie.lines!(plot, horizons, beta; color = linecolor, linewidth = linewidth)

    # Zero line
    if drawzero
        Makie.hlines!(plot, [0.0]; color = zerolinecolor, linewidth = 1, linestyle = :dash)
    end

    return plot
end

# ============================================================================
# Convenience wrappers mapping irfplot/irfplot! to irfplotmakie/irfplotmakie!
# ============================================================================

irfplot(lp::AbstractLocalProjection; kwargs...) = irfplotmakie(lp; kwargs...)
irfplot(lp::AbstractLocalProjection, est; kwargs...) = irfplotmakie(lp + vcov(est); kwargs...)
irfplot!(ax, lp::AbstractLocalProjection; kwargs...) = irfplotmakie!(ax, lp; kwargs...)
function irfplot!(ax, lp::AbstractLocalProjection, est; kwargs...)
    return irfplotmakie!(ax, lp + vcov(est); kwargs...)
end

function irfplot_axis(subfig, lp, est; kwargs...)
    return irfplot_axis(subfig, lp + vcov(est); kwargs...)
end

function irfplot_axis(subfig, lp; kwargs...)
    kw = Dict{Symbol, Any}(kwargs)
    term = get(kw, :term, lp.shock)
    title = pop!(kw, :title, "")
    xtickstep = pop!(kw, :xtickstep, 4)
    horizons = collect(0:lp.horizon)
    xticks_val = xtickstep > 0 ? (horizons[1]:xtickstep:horizons[end]) : Makie.automatic
    ax = Makie.Axis(
        subfig[1, 1];
        xlabel = "Horizon",
        ylabel = string(term),
        title = title,
        xticks = xticks_val
    )
    Makie.xlims!(ax, horizons[1] - 0.4, horizons[end] + 0.4)
    p = irfplotmakie!(ax, lp; kw...)
    return (subfig, ax, p)
end

end # module
