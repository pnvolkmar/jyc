# Pkg.clone("https://github.com/davidavdav/CHull.jl", "CHull")using MathProgBase
using Gurobi
using PyCall
using CHull
using PyPlot
using Distributions
using Ipopt
using PyCall

#####################
# hulls and verticies
######################

@pyimport scipy.spatial as spatial

# try
#     type Chull{T<:Real}
#         points::Array{Array{T,1},1}
#         vertices::Array{Int,1}
#         simplices::Array{Any,1}
#     end
# catch
# end

# function chull{T<:Real}(x::Array{T})
#     py = spatial.ConvexHull(x)
#     points = convert(Array{Array{T,1},1},py["points"])
#     vertices = convert(Array{Int},py["vertices"]) + 1
#     simplices = convert(Array{Array{Int,1},1},py["simplices"]) + 1
#     Chull(points, vertices, simplices)
# end

#####################################################
# INPUTS
#####################################################

tol   = .1 # tolerance
delta = .8  # Setting discount factor
DataDir = "/Users/petervolkmar/Dropbox/JYC/Code/version-control"
cd(DataDir)
# DataDir = "C:\\Users\\pnvolkmar\\Dropbox\\JYC\\Code\\"
# include(DataDir * "LNGdata" * string(year) * ".jl")
include("jyc_ex_data.jl")
players = [p0, p0, p0]


# function jyc_ex(players)

chart_title = "Cournot ex w"
for i in 1:length(players)
    if i == length(players); chart_title *= " and ";
    elseif i > 1; chart_title *= ", ";
    else; chart_title *= " "; end;
    chart_title *= players[i].Name
end

#####################################################
# Settings
#####################################################

# Technical setting for approximation
del1 = 1 - delta
n    = 8 # number of search gradients
M    = 4 # number of points
N    = length(players) # number of players
cen  = ones(Float32, 1, N) ####### THIS IS WHERE THE
#####################################################
# Hausdorff Distance - not always used...generally runs more slowly.
#####################################################

function d(a, b)
	sqrt(sum((a.-b).^2))
end

function hausdorff(A, B)
	C = maximum([minimum([d(A[j,:],B[i,:]) for i = 1:size(B,1)]) for j = 1:size(A,1)])
	D = maximum([minimum([d(A[i,:],B[j,:]) for i = 1:size(A,1)]) for j = 1:size(B,1)])
    if C >= D
    	return C
    else
    	return D
    end
end

#####################################################
# Building Vectors
#####################################################
tic();
# Build a list if all possible player actions A = A_1 x A_2 x ... x A_N
actions = Array{Float32}(1,N);
for (i,player) in enumerate(players)
    actions = repeat(actions,inner = [M,1])
    for idx = 1:size(actions,1)
        q = player.Max*((idx-1)%M)/(M-1)
        actions[idx,i] = q
    end
end

# List of stage game payoffs for each action
stagepay = Array{Float32}(size(actions));
for (p, player) in enumerate(players)
    for i = 1:size(actions,1)
        # print(i); print(p); print(player); print("\n")
        q = actions[i,p]
        Q = sum(actions[i,:])
        c = player.Cost(q)
        stagepay[i,p] = q*(P(Q))-c
    end
end

# Best Response function
BR = copy(stagepay);
for p = 1:N
    for j = 0:(M^(N-p)-1)
        for i = 1:(M^(p-1))
        	#println(i+j*M^p,":",M^(p-1),":",i +2*M^(p-1)+ j*M^(p)," , ",N-p+1," ", p,j,i)
        	a = stagepay[i+j*M^p : M^(p-1) : i+(M-1)*M^(p-1)+j*M^p , N-p+1]
        	BR[i+j*M^p : M^(p-1) : i+(M-1)*M^(p-1)+j*M^p , N-p+1] = maximum(a)
        end
    end
end

#####################################################
# Setting up search alogoritms
#####################################################


H = randn(Float32, (n, N))
H = H./ (sqrt(sum(H.^2, 2))*ones(Float16, 1, N)
# H = readdlm("H.csv",',',Float64) #THIS IS FOR COMPARISON PURPOSES ONLY.
rad = Float32(ceil(maximum(stagepay)))


Z = H*rad.+ones(Float32, n, 1)*cen

println("\n\nSPNE with $(N) Players")
println("Outer Approximation")
C=sum(Z.*H,2)
L=n # this should really just be the number of gradient specified as 'n' above.
G=copy(H) #Use subgradient same as search directions
wmin=minimum(BR,1)'
iter=0
tolZ=20.0
tolZH = 20.0
Zold=zeros(size(Z))
rtouter = [tolZ]
rtouterH = [tolZ]

# function B_outer(stagepay,L,N,H, delta, C, BR,Z)
while tolZH < tol
    del1 = 1-delta
    len_stagepay = size(stagepay,1)
    Cla=zeros(L,len_stagepay)
    Wla=zeros(N,L,len_stagepay)
    for l = 1:L
        for a = 1:len_stagepay
            pay = stagepay[a,:]''
            env = Gurobi.Env()
            setparam!(env,"OutputFlag",0)
            model = gurobi_model(env; name = "lp_01",
                f = vec(-H[l,:]),
                A = [H;-eye(N)],
                b = vec([delta*C+del1*H*pay;-del1*BR[a,:]''-delta*wmin]))
            optimize(model);
            #if no optimum is found, then cla=-inf
            if get_status(model)!=:optimal
                Cla[l,a]=-10000000000
            else
                Wla[:,l,a] = get_solution(model)
                Cla[l,a]= -get_objval(model)
            end
            # env = 0; gc(); # do we need more memory clean up?
        end
    end
    # Couter = copy(C)
    C = maximum(Cla,2)
    I = mapslices(indmax, Cla, 2)[:]

    for l = 1:L
        Z[l,:]=Wla[:,l,I[l]]
    end
    wmin=minimum(Z,1)'
    # Convergence
    tolZ  = maximum(abs(Z-Zold)./(1+abs(Zold)))
    tolZH = hausdorff(Z, Zold)
    # push!(rtouter,tolZ)
    # push!(rtouterH, tolZH)

    if iter%5 == 0
        @printf("iteration: %d \t tolerance: %.2f \n", iter, tolZH)
    end
    Zold=copy(Z)
    iter=iter+1
end
    # gc()
toc()
@printf("Convergence after %d iterations. \n \n", iter)

outerpts = copy(Z)

#   # ### # ### #     #    ##########################
## ##  ## #  ## # ##### ### ###########################
## ## # # # # # #   ###    ##############################
## ## ##  # ##  # ##### #  ############################
#   # ### # ### #     # ##  #########################
print("Inner Approximation\n")

tic()

#####################################################
# Setting up search alogoritms
#####################################################
delrat = del1/delta

cen = mean([mean(outerpts,1);minimum(outerpts,1)],1)
# cen = minimum(outerpts,1)
rad = minimum(mean(outerpts,1)-minimum(outerpts,1))/4

# Gradients and Tangency Points
Z = H*rad.+ones(n,1)*cen

#####################################################
# Running the loop - this code taken almost entirely from Yeltekin
#####################################################

C = sum(Z.*H,2)
G = H

# Parameters of Iteration
wmin=minimum(BR,1)'
iter = 0
tolZ = 20.0
tolZH = 20.0
Zold = zeros(size(Z))
len_stagepay = size(stagepay,1)
rtinner = [tolZ]
rtinnerH = [tolZ]

while tolZH > tol
    # Optimization
    # Construct Iteration
    Cla=zeros(L,len_stagepay)
    Zla=zeros(N,L,len_stagepay)
# if size(Z,2) > 2
#     ax[:scatter](Z[:,1], Z[:,2], Z[:,3], zdir="z", c=c=[1-(iter%20)/20,0.1,(iter%20)/20])
# else
#     ax[:scatter](Z[:,1], Z[:,2], c=[1-(iter%20)/20,0.1,(iter%20)/20])
# end
    for l = 1:L
        for a = 1:len_stagepay
            pay = stagepay[a,:]''
            env = Gurobi.Env()
            setparam!(env,"OutputFlag",0)
            model = gurobi_model(env; name = "lp_inner",
                f = vec(-H[l,:]),
                A = [G;-eye(N)],
                b = vec([delta*C+del1*G*pay;-del1*BR[a,:]''-delta*wmin]))
            optimize(model);
            # if no optimum is found, then cla = -inf
            if get_status(model) !=:optimal
                Cla[l,a] = -10000000000
            else
                Zla[:,l,a] =get_solution(model)
                Cla[l,a]=-get_objval(model)
            end
        end
    end
    # Cinner = copy(C)
    C = maximum(Cla,2)
    I = mapslices(indmax, Cla, 2)[:]

    for l = 1:L
        Z[l,:]=Zla[:,l,I[l]]
    end

    if maximum(isnan(Z)) == true
        println("Convergence failed with this Z:")
        println(Z[0,:])
        println("This is the source of the trouble")
        Zinner = copy(Zold)
        println(Zinner[0,:])
        println("along with this C")
        println(Cinner[0,:])
        wmininner = copy(wmin)
        Z = Zinner
        tolZ, tolZH = tol, tol #just fail out of this loop
    end

    wmin = minimum(Z,1)'
    # Convergence
    tolZ  = maximum(abs(Z-Zold)./(1+abs(Zold)))
    tolZH = hausdorff(Z, Zold)
    push!(rtinner, tolZ)
    push!(rtinnerH, tolZH)

    if iter%5 == 0
        @printf("iteration: %d \t tolerance: %.2f \n", iter, tolZH)
    end
    Zold = copy(Z)
    iter = iter + 1
end
#####################################################
# OUTPUTS
#####################################################

toc()
@printf("Convergence after %d iterations. \n \n", iter)
innerpts=copy(Z)

k_inner = chull(innerpts);
k_outer = chull(outerpts);

##   ##    #### ###    ### ### #################################
# ##### ### ## # ## ### ## ### #################################
# #   #    ##     #    ###     #################################
# ## ## #  ## ### # ###### ### #################################
##   ## ##  # ### # ###### ### #################################

if size(Z,2) > 2
    fig = figure()
    ax  = fig[:gca](projection="3d")
else
    fig, ax = subplots()
end

p3 = 3
p2 = 2
if size(Z,2) > 2
    ax[:scatter](stagepay[:,1], stagepay[:,p2], stagepay[:,p3], zdir="z", c="b")
    for simplex in k_inner.simplices
        ax[:plot](innerpts[simplex, 1], innerpts[simplex, p2], innerpts[simplex, p3], "g--")
    end
    # ax[:scatter](outerpts[:,1], outerpts[:,p2], outerpts[:,p3], zdir="z", c="y")
    for simplex in k_outer.simplices
        ax[:plot](outerpts[simplex, 1], outerpts[simplex, p2], outerpts[simplex, p3], "y--")
    end
    ax[:set_zlabel](players[p3].Name)
else
    for simplex in k_inner.simplices
        ax[:plot](innerpts[simplex, 1], innerpts[simplex, p2], "g-")
    end
    for simplex in k_outer.simplices
        ax[:plot](outerpts[simplex, 1], outerpts[simplex, p2], "y--")
    end
end


ax[:set_title](chart_title)
ax[:set_xlabel]("Payoff to Player 1")
ax[:set_ylabel]("Payoff to Player 2")

# Mapping current production on to the value set:

max   = [player.Max for player in players]
Q_max = sum(max)
val_max = max * P(Q_max)-[players[i].Cost(max[i]) for i in 1:N]

# if size(Z,2) > 2
#     ax[:scatter](val_max[1], val_max[p2], val_max[p3], zdir="z", c="r")#, marker = "v")
# else
#     ax[:scatter](val_max[1], val_max[p2], c="r")
# end
# end # for the function jyc_ex


# jyc_ex([p0,p0])
# jyc_ex([p0,p0,p0])
# jyc_ex([p0,p0,p0])
# jyc_ex([p0,p0,p0,p0])
# jyc_ex([p0,p0,p0,p0])
# jyc_ex([p0,p0,p0,p0,p0])
# jyc_ex([p0,p0,p0,p0,p0,p0])

##################################
# Analysis
##################################

using JuMP
function in_hull(x, Z)
    (R,C)=size(Z)
    m = Model()
    @variable(m, 0 <= lambda[j=1:R] <= 1)
    @constraint(m, inhull[i=1:C], x[i] == sum{Z[j,i]*lambda[j], j = 1:R})
    @constraint(m, sum(lambda) == 1)
    status = solve(m)
    if status == :Optimal
        return(true)
    else
        return(false)
    end
end

function max_points(Z,G)
    (R,C)=size(Z)
    m = Model(solver=IpoptSolver())
    @variable(m, z[i=1:C])
    @constraint(m, inhull[i=1:R], sum{G[i,j]*z[j], j=1:C} <= sum{G[i,j]*Z[i,j], j=1:C})
    @NLobjective(m, Max, sum{z[j]^2, j=1:C})
    solve(m)
    println("z = ", getvalue(z))
end
