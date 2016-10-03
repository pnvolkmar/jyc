type Player
    Name::String
    Production::Float64
    Max::Float64
    Cost::Function
end

p0     = Player("Player Cost=0.0", 5, 6, q->q*0)
p6     = Player("Player Cost=0.6", 5, 6, q->q*0.6)
