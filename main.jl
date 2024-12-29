using CSV
using DataFrames
using Statistics
include("src/utils.jl")
include("src/model.jl")
include("src/train.jl")

# Load and preprocess data
X, Y = load_data("data/pendulum_data.csv")
X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

X_train = (X_train .- mean(X_train)) ./ std(X_train)
X_test = (X_test .- mean(X_test)) ./ std(X_test)

# Convert DataFrames to Matrices and transpose
X_train = Matrix(X_train)'
Y_train = Matrix(Y_train)'
X_test = Matrix(X_test)'
Y_test = Matrix(Y_test)'

X_train = Float32.(X_train)
Y_train = Float32.(Y_train)
X_test = Float32.(X_test)
Y_test = Float32.(Y_test)

# Check data shapes
println("X_train size: ", size(X_train))
println("Y_train size: ", size(Y_train))

# Create and train the model
model = create_model()
train_model(model, X_train, Y_train, X_test, Y_test, 50, 0.001)
