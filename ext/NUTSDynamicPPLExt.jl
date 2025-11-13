module NUTSDynamicPPLExt

using NUTS: NUTS
using DynamicPPL
# DI = DifferentiationInterface
using DynamicPPL: AbstractMCMC, DI, Random

struct FastNUTSState{L<:DynamicPPL.Experimental.FastLDF}
    ldf::L
    position::Vector{Float64}
end

# LogDensityProblems.jl doesn't provide an in-place function, so we have to use DI directly.
# Unfortunately this also means having to take on a DynamicPPL dependency. Otherwise we
# could just use Turing's external sampler mechanism.
function NUTS.log_density_gradient!(ldf::DynamicPPL.Experimental.FastLDF, x::AbstractVector, grad::AbstractVector)
    return first(DI.value_and_gradient!(
        DynamicPPL.Experimental.FastLogDensityAt(
            ldf.model, ldf._getlogdensity, ldf._iden_varname_ranges, ldf._varname_ranges
        ),
        grad,
        ldf._adprep,
        ldf.adtype,
        x,
    ))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::NUTS.FastNUTS;
    kwargs...
)
    # Create a LDF
    vi = DynamicPPL.link!!(DynamicPPL.VarInfo(rng, model), model)
    ldf = DynamicPPL.Experimental.FastLDF(model, DynamicPPL.getlogjoint_internal, vi; spl.adtype)
    # Get initial parameters
    # TODO(penelopeysm): These could obviously be more sophisticated, right now these
    # parameters are just sampled from the prior, because that's what `VarInfo(rng, model)`
    # does.
    initial_params = vi[:]
    transition = DynamicPPL.ParamsWithStats(initial_params, ldf)
    state = FastNUTSState(ldf, initial_params)
    return (transition, state)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::NUTS.FastNUTS,
    state::FastNUTSState;
    kwargs...
)
    # Take a step
    nt_state = (rng=rng, posterior=state.ldf, stepsize=spl.stepsize, position=state.position)
    nt_state = NUTS.nuts!!(nt_state)
    # `nt_state.position` will now contain the new position
    params_vector = nt_state.position
    transition = DynamicPPL.ParamsWithStats(params_vector, state.ldf)
    state = FastNUTSState(state.ldf, params_vector)
    return (transition, state)
end

end # module
