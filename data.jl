type Player
    Name::String
    Production::Float64
    Max::Float64
    Cost::Function
end

data = readdlm("test.txt")

for i  in 2:(size(data,1))
    assignment = parse("$(data[i,1]) = Player(\"$(data[i,1])\", $(data[i,2]), $(data[i,3]), $(data[i,4]))")
    eval(assignment)
end

function piecewise(x::Symbol,c::Expr,f::Expr)
  n=length(f.args)
  @assert n==length(c.args)
  @assert c.head==:vect
  @assert f.head==:vect
  vf=Vector{Function}(n)
  for i in 1:n
    vf[i]=@eval $x->$(f.args[i])
  end
  return @eval ($x)->($(vf)[findfirst($c)])($x)
end

# updating KSA's cost function
Saudi_Arabia.Cost = piecewise(:x, :([x < 6, x >= 6]), :([4.5*x, 4.5*6+12.0*(x-6)]))
