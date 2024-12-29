using Flux

# Define the neural network model
function create_model()
    return Chain(
        Dense(2, 32, relu),
        Dense(32, 32, relu),
        Dense(32, 2)  # Output: θ_next, ω_next
    )
end