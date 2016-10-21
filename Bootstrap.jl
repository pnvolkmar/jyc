# Pkg.add("PyCall")
# Pkg.clone("https://github.com/davidavdav/CHull.jl.git")
# Pkg.add("PyPlot")
# http://stackoverflow.com/questions/26080202/memory-usage-blows-up-when-iterating-over-array
# http://docs.julialang.org/en/release-0.4/manual/performance-tips/
using MathProgBase
using Gurobi
using PyCall
using CHull
using PyPlot
using Distributions
DataDir = "/Users/petervolkmar/Dropbox/JYC/Code"
cd(DataDir)
include("data.jl")

#####################################################
# INPUTS
#####################################################

n     = 30  # # of gradients
tol   = 1.0 # tolerance
delta = .6  # Setting discount factor
players = [Saudi_Arabia, Iraq, UAE]
chart_title = "Estimate for"
for i in 1:length(players)
    if i == length(players); chart_title *= " and ";
    elseif i > 1; chart_title *= ", ";
    else; chart_title *= " "; end;

    chart_title *= players[i].Name
end
srand(1234)

#####################################################
# Settings
#####################################################
tic()

# Technical setting for outer approximation
M    = 4 # number of points
N    = length(players)
cen  = ones(1,N) ####################################### THIS IS WHERE THE CENTER IS

del1 = 1-delta

# Setting price function
A = randn() * 226.882 + 341.418
B = randn() * 2.420 - 3.075
production = 0.0
for i = 1:N
    production = production + players[i].Production
end

function P(Q)
	price = A-B*(Q+93.8-production)
	if price > 0
		return price
	else
		return 0.0
	end
end

#####################################################
# Hausdorff Distance - not always used...generally runs more slowly.
#####################################################

function d(a, b)
	sqrt(sum((a.-b).^2))
end

function hausdorff(A, B)
	C = maximum([minimum([d(A[j,:],B[i,:]) for i = 1:size(B,2)]) for j = 1:size(A,2)])
	D = maximum([minimum([d(A[i,:],B[j,:]) for i = 1:size(A,2)]) for j = 1:size(B,2)])
    if C >= D
    	return C
    else
    	return D
    end
end

#####################################################
# Building Vectors
#####################################################
tic()
# Build a list if all possible player actions A = A_1 x A_2 x ... x A_N
actions = zeros(1,N)
for (i,player) in enumerate(players)
# i, player = 1, Angola
    actions = repeat(actions,inner = [M,1])
    for idx = 1:size(actions,1)
        q = player.Max*((idx-1)%M)/(M-1)
        actions[idx,i] = q
    end
end

# List of stage game payoffs for each action
stagepay = zeros(size(actions))
for (p, player) in enumerate(players)
    for i = 1:size(actions,1)
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

H = randn(n, N) # H will be the gradients
H = H./ (sqrt(sum(H.^2, 2))*ones(1,N))
# H = readdlm("H.csv",',',Float64) #THIS IS FOR COMPARISON PURPOSES ONLY.
rad = ceil(maximum(stagepay))
Z = H*rad.+ones(n,1)*cen

println("Outer Approximation")
C=sum(Z.*H,2)
L=n # this should really just be the number of gradient specified as 'n' above.
G=H #Use subgradient same as search directions
wmin=minimum(BR,1)'
iter=0
tolZ=20.0
tolZH = 20.0
Zold=zeros(size(Z))
len_stagepay = size(stagepay,1)
rtouter = [tolZ]
rtouterH = [tolZ]

while tolZH>tol
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
                b = vec([delta*C+del1*H*pay;-del1*BR[a,:]-delta*wmin]))
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

    # if maximum(isnan(Z)) == true
    #     println("This is the trouble Z")
    #     println(Z[1,:])
    #     println("This is the source of the trouble")
    #     Zouter = copy(Zold)
    #     println(Zouter[1,:])
    #     println("along with this C")
    #     println(Couter[1,:])
    #     wminouter = copy(wmin)
    #     Z = Zouter
    #     tolZ, tolZH = tol, tol # just fail out of this loop
    # end

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
    # gc()
end
toc()
@printf("Convergence after %d iterations. \n \n", iter)

outerpts = copy(Z)

#   # ### # ### #     #    #####################################
## ##  ## #  ## # ##### ### ####################################
## ## # # # # # #   ###    #####################################
## ## ##  # ##  # ##### #  #####################################
#   # ### # ### #     # ##  ####################################
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
                b = vec([delta*C+del1*G*pay;-del1*BR[a,:]-delta*wmin]))
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

# println("First 5 inner points are:")
# println(innerpts[1:5,:])
# println("First 5 outer points are:")
# println(outerpts[1:5,:])
# using PyCall
# @pyimport scipy.spatial as spatial
#
# try
#     type Chull{T<:Real}
#         points::Array{Array{T,1},1}
#         vertices::Array{Int,1}
#         simplices::Array{Any,1}
#     end
# catch
# end
# import CHull.chull
# function convex_hull{T<:Real}(x::Array{T})
#     py = spatial.ConvexHull(x)
#     points = convert(Array{Array{T,1},1},py["points"])
#     vertices = convert(Array{Int},py["vertices"]) + 1
#     simplices = convert(Array{Any,1},py["simplices"]) + 1
#     Chull(points, vertices, simplices)
# end

# function show(ch::Chull)
#     println(string("Convex Hull of ", size(ch.points,1), " points in ", size(ch.points[1],1), " dimensions"))
#     println("Hull segment vertex indices:")
#     println(ch.vertices)
#     println("Points on convex hull in original order:\n")
#     println(ch.points[sort(ch.vertices[:,1]),:])
# end

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
if size(Z,2) > 2
    ax[:scatter](stagepay[:,1], stagepay[:,2], stagepay[:,3], zdir="z", c="r", marker="+")
    for simplex in k_inner.simplices
        ax[:plot](innerpts[simplex, 1], innerpts[simplex, 2], innerpts[simplex, 3], "y-")
    end
    # ax[:scatter](outerpts[:,1], outerpts[:,2], outerpts[:,3], zdir="z", c="y")
    for simplex in k_outer.simplices
        ax[:plot](outerpts[simplex, 1], outerpts[simplex, 2], outerpts[simplex, 3], "g-")
    end
    ax[:set_zlabel](players[3].Name)
    ax[:set_title]("Value Set of 3 Players")
else
    ax[:scatter](innerpts[:,1], innerpts[:,2], c="k")
    for simplex in k_inner.simplices
        ax[:plot](innerpts[simplex, 1], innerpts[simplex, 2], "g-")
    end
    ax[:scatter](outerpts[:,1], outerpts[:,2], c="k")
    for simplex in k_outer.simplices
        ax[:plot](outerpts[simplex, 1], outerpts[simplex, 2], "g--")
    end
    ax[:set_title]("Value Set of Both Players")
end


ax[:set_xlabel](players[1].Name)
ax[:set_ylabel](players[2].Name)

# Mapping current produciton on to the value set:

current   = [player.Production for player in players]
Q_current = sum(current)
val_current = current * P(Q_current)-[players[i].Cost(current[i]) for i in 1:N]

max   = [player.Max for player in players]
Q_max = sum(max)
val_max = max * P(Q_max)-[players[i].Cost(max[i]) for i in 1:N]

if size(Z,2) > 2
    ax[:scatter](val_current[1], val_current[2], val_current[3], zdir="z", c="k")
    ax[:scatter](val_max[1], val_max[2], val_max[3], zdir="z", c="r")
else
    ax[:scatter](val_current[1], val_current[2], c="k", marker = '^')
    ax[:scatter](val_max[1], val_max[2], c="r", marker = '^')
end

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
# Infeasible means not in the hull, Optimal means in the hull

# good = np.sum(in_hull(innerpts,outerpts))
# println('\n'+str(good)+' of '+str(L)+' inner approx points are within the outer approx.')

# # profits2015 = np.ones(N)
# # actions2015 = np.array([1.80,.54,10.19])
# # for p, player in enumerate(players):
# #     q = actions2015[p]
# #     Q = sum(actions2015)
# #     c = player.Cost(q)
# #     profits2015[p] = q*(P(Q))-c

# # ax1.scatter(profits2015[0],profits2015[1],profits2015[2], c='g')
# # plt.title('Current and hypothetical production: ' + players[0].Name + ' ' \
# #     + players[1].Name + " and " + players[2].Name)

# ax1.set_xlabel(players[0].Name)
# ax1.set_ylabel(players[1].Name)
# ax1.set_zlabel(players[2].Name)
# plt.title('Inner approx is blue')
# plt.show()
