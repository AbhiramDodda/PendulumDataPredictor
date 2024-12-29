using Flux
using Flux.Optimise: Adam
using Plots

function train_model(
    model::Chain,
    X_train::Matrix{Float64},
    Y_train::Matrix{Float64},
    X_test::Matrix{Float64},
    Y_test::Matrix{Float64},
    epochs::Int,
    lr::Float64
)
    loss_fn(m, x, y) = Flux.mse(m(x), y)  # Define loss function
    optimizer = Flux.setup(Adam(lr), model)  # Setup optimizer

    train_loss = Float64[]
    test_loss = Float64[]

    opt_state = Flux.setup(Flux.Optimiser.Descent(lr), model)
    for epoch in 1:epochs
        for (x, y) in zip(X_train, Y_train)
            grads = Flux.gradient(() -> Flux.Losses.mse(model(x), y), model)
            Flux.update!(opt_state, model, grads)
        end
        train_loss = Flux.Losses.mse(model(X_train), Y_train)
        test_loss = Flux.Losses.mse(model(X_test), Y_test)
        println("Epoch $epoch | Train Loss: $train_loss | Test Loss: $test_loss")
    end


    # Plot Loss Curves
    plot(1:epochs, train_loss, label="Train Loss")
    plot!(1:epochs, test_loss, label="Test Loss", xlabel="Epochs", ylabel="Loss")
end