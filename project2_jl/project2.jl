#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#

# Example:
using LinearAlgebra

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")


"""
    optimize(f, g, c, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `c`: Constraint function for 'f'
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, c, x0, n, prob)

    # # Try hyper Nesterov Momentum
    # α = 0.1
    # β = 0.5
    # x = copy(x0)
    # x_history = copy(x0)
    # v = zeros(length(x))
    # g_curr = zeros(length(x))
    # g_prev = zeros(length(x))
    # μ = 0.005
    # while count(f,g) < n
    #     g_curr = normalize!(copy(g(x)))
    #     α = α - μ*(g_curr⋅(-g_prev - β*v))
    #     v[:] = β*v + g_curr
    #     g_prev = copy(g_curr)
    #     x = x - α*(g_curr + β*v)
    #     x_history = vcat(x_history, x)
    #     # println(α)
    #     # # print(x)
    #     # print(f(x))
    # end
    #
    # # TODO: Create separate file and function that will run optimizer with 3 different
    # # random seeds, and then generate the plots.
    # # see localtest.jl for tips
    #
    #
    # # TODO if time:
    # # Try generalized pattern search
    # # Try Nelder-mead
    #
    # x_best = x
    # return x_best
    x_best = x0
    return x_best
end
