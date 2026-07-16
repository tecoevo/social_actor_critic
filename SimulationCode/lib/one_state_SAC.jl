function SAC_unbiased_1_state!(du, u, p, t)
    x, s = u
    ω, α, β, full_demonstrator_choice = p
    ω_SL = full_demonstrator_choice ? ω*(1-ω) : ω
    du[1] = (1-ω)*α*( x^2*(1-x)^2 + s*x*(1-6x+6x^2) )
    du[2] = (1-ω)*( 4*α*s*(1-x)*x*(1-2x) ) + ω_SL*(β^2*(1-x)*x - (2-β)*β*s)
    # du[2] = (1-ω)*( 4*α*s*(1-x)*x*(1-2*x) + α^2*x*(1-x)*(x^2*(1-3*x+2*x^2)^2 + s*(3 + x*(1-x)*(112*x*(1-x) - 39))) ) + ω*(β^2*(1-x)*x - (2-β)*β*s)
    return nothing
end

function SAC_perfbiased_1_state!(du, u, p, t)
    x, s = u
    ω, α, β, full_demonstrator_choice = p
    ω_SL = full_demonstrator_choice ? ω*(1-ω) : ω
    du[1] = (1-ω)*α*( x^2*(1-x)^2 + s*x*(1-6x+6x^2) )  + ω_SL*(β*s/x)
    du[2] = (1-ω)*( 4*α*s*(1-x)*x*(1-2x) ) + ω_SL*(β^2*(1-x)*(s+x^2)-2*β*s*x)/x
    return nothing
end