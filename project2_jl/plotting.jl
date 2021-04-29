using ImplicitEquations, Plots

include("project2.jl")
include("helpers.jl")

probname = "simple1"
n_seeds = 3
seed = 50
num_iter = 30

prob = PROBS[probname]

x0 = prob.x0()

# TODO: Contour plot for Rosenbrock function:
# use range of (-3.0, 3.0) for x1, x2
# z(x1, x2) = f([x1, x2])
x1 = range(-3.0,stop=3.0,length=50)
x2 = range(-3.0,stop=3.0,length=50)
funcn = zeros(50,50)
# c1 = zeros(50,50)
# c2 = zeros(50,50)
for i = 1:50
    for j = 1:50
        global funcn
        funcn[i,j] = prob.f([x1[i], x2[j]])
        # c1[i,j] = x1[i] + x2[j]^2 - 1
        # c2[i,j] = -x1[i] -x2[j]
    end
end

c1(x, y) = x + y^2 - 1
c2(x, y) = -x - y
p1 = contour(x1, x2, funcn, colorbar = false,
    c = cgrad(:viridis, rev = true), legend = false, xlims = (-3, 3), ylims = (-3, 3),
    xlabel = "x₁", ylabel = "x₂", aspectratio = :equal)
plot!(c2 ≤ 0)
# contour!(x1, x2, c1, levels = [0], c = :red, fillrange(0,0))
# contour!(x1, x2, c2, levels = [0], c = :black)
# contour!(x1, x2, c1, levels = [0], c = :gray)
# plot!(xhist_nseeds[:,1,1], xhist_nseeds[:,2,1])
# plot!(xhist_nseeds[:,1,2], xhist_nseeds[:,2,2])
# plot!(xhist_nseeds[:,1,3], xhist_nseeds[:,2,3])
# savefig(p1, "rosenbrock_trajectories.pdf")


# Plot x_history on top of function contour


# # CONVERGENCE PLOTS
# fhist_nseeds = zeros(num_iter, n_seeds)
#
# for i = 1:n_seeds
#     Random.seed!(seed+i)
#     _, fhist_nseeds[:,i] = optimize_plot_data(prob.f, prob.g, prob.x0(), prob.n, probname)
# end
#
# p = plot(1:num_iter, fhist_nseeds, xlabel = "Iterations", ylabel = "Function value", title = "Rosenbrock Convergence Plot", label = ["Seed 1" "Seed 2" "Seed 3"], lw = 2)
# savefig(p, "rosenbrock_convergence.pdf")
