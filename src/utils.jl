using CSV
using DataFrames

# Load data from CSV
function load_data(file_path)
    df = CSV.read(file_path, DataFrame)
    X = hcat(df.θ, df.ω)
    Y = hcat(df.θ_next, df.ω_next)
    return X, Y
end

# Split data into train/test sets
function train_test_split(X, Y, train_ratio=0.8)
    n = size(X, 1)
    idx = shuffle(1:n)
    train_size = Int(train_ratio * n)
    train_idx, test_idx = idx[1:train_size], idx[train_size+1:end]
    return X[train_idx, :], Y[train_idx, :], X[test_idx, :], Y[test_idx, :]
end