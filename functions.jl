
#####################################################
# Hausdorff Distance
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

##################################
# Analysis
##################################

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

##################################
# B Outer
##################################

function B_o(
    stagepay,
    BR,
    H,
    Z,
    C,
    wmin,
    L::Int,
    N::Int,
    delta::Float64,
    )
    del1 = 1-delta
    len_stagepay = size(stagepay,1)
    local Cla=zeros(L,len_stagepay)
    local Wla=zeros(N,L,len_stagepay)
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
    C = maximum(Cla,2)
    I = mapslices(indmax, Cla, 2)[:]
    for l = 1:L
        Z[l,:]=Wla[:,l,I[l]]
    end
    wmin=minimum(Z,1)'
    return H,Z,C,wmin
end
