using AbstractMCMC: AbstractMCMC

# We have to define this struct here, since structs in extensions can't be loaded. Even
# though the `adtype` field should really have a more concrete types, we avoid specifying it
# here to avoid taking on a hard dependency on ADTypes.jl
struct FastNUTS{AD} <: AbstractMCMC.AbstractSampler
    stepsize::Float64
    adtype::AD # should be ADTypes.AbstractADType
end
