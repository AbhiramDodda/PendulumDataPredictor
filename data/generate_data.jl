using CSV
using DataFrames
using Random

function generate_pendulum_data(samples::Int, time_step::Float64)
    θ = π / 4  # Initial angle (radians)
    ω = 0.0    # Initial angular velocity (rad/s)
    g = 9.81   # Gravitational acceleration (m/s²)
    l = 1.0    # Length of pendulum (meters)

    data = DataFrame(θ=Float64[], ω=Float64[], θ_next=Float64[], ω_next=Float64[])
    for _ in 1:samples
        α = -(g / l) * sin(θ)  # Angular acceleration
        θ_next = θ + ω * time_step
        ω_next = ω + α * time_step

        push!(data, (θ, ω, θ_next, ω_next))

        θ = θ_next + 0.01 * randn()  # Add noise
        ω = ω_next + 0.01 * randn()  # Add noise
    end
    return data
end

function save_data_to_csv(data, file_path)
    CSV.write(file_path, data)
end

# Generate and save data
data = generate_pendulum_data(10000, 0.01)
save_data_to_csv(data, "data/pendulum_data.csv")