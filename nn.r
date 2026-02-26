#dC/dw_ij = dC/dV_j*dV_j/dw_ij = dC/dV_j*act(V_i), where act is the
#activation function and V is the value of node i(pre activation)
#dC/dV_i = dC/d(act(V_i))*d(act(V_i))/dV_i
#dC/d(act(V_i)) = sum(dC/dV_j*dV_j/dV_i, j) = sum(dC/dV_j*w_ij, j)
#dC/db_i = dC/dV_i*dV_i/db_i = dC/dV_i

#Constants
n <- 3 #number of nodes in each column
m <- 3 #number of cols
i <- 1 #number of inputs
o <- 1 #number of outputs

# nn_weights should be a random sample
nn_seed <- rep(x = 1, times = m * n * n - n * n)
nn_weights <- array(nn_seed, dim = c(m - 1, n, n))

#these should also be random samples but wtvr
in_weights <- array(1, dim = c(n, i))
out_weights <- array(1, dim = c(o, n))
out_biases <- array(1, dim = o)

#bias
biases <- array(1, dim = c(n, m))

act <- function(input) {
    #input is a vector(doesn't rly matter dimension)
    return(pmax(0, input)) # relu
}
dact <- function(input) {
    as.integer(input > 0)
}
loss <- function(input) {
    #loss function, usually mean squared or smth
    #using mean squared bc the total loss is j the derivative is nice
    sum(input^2)
}
dloss <- function(input) {
    2 * input
}


model <- list(
    dim = c(n = n, m = m, i = i, o = o),
    weights = list(hidden = nn_weights, input = in_weights, out = out_weights),
    biases = list(hidden = biases, out = out_biases),
    act = list(hidden = act, out = act),
    dact = list(hidden = dact, out = dact)
) #makes it easier to move around

run_net <- function(input, model) {
    n <- model$dim[["n"]]
    m <- model$dim[["m"]]
    o <- model$dim[["o"]]
    values <- matrix(nrow = n, ncol = m)
    values[, 1] <- model$weights$input %*% input + model$biases$hidden[, 1]
    for (i in 2:m) {
        values[, i] <- model$weights$hidden[i - 1, , ] %*%
            model$act$hidden(values[, i - 1]) +
            model$biases$hidden[, i]
    }
    dim(model$act$hidden(values[, m]))

    w_out <- matrix(as.numeric(model$weights$out), nrow = o, ncol = n)
    h_m <- matrix(as.numeric(model$act$hidden(values[, m])), nrow = n, ncol = 1)
    b_out <- matrix(as.numeric(model$biases$out), nrow = o, ncol = 1)

    out <- w_out %*% h_m + b_out
    # out <- model$weights$out %*% matrix(model$act$hidden(values[, m]), ncol = 1) + model$biases$out
    list(input = input, output = out, model = model, values = values)
}

input <- c(3)
output <- run_net(input, model)
correct_output <- matrix(1, 1, 1)

backprop_args <- list(
    output = output$output,
    correct_output = correct_output,
    values = output$values,
    model = model,
    dloss = dloss,
    input = input
)

#dC/dw_ij = dC/dV_j*dV_j/dw_ij = dC/dV_j*act(V_i), where act is the
#activation function and V is the value of node i(pre activation)
#dC/dV_i = dC/d(act(V_i))*d(act(V_i))/dV_i
#dC/d(act(V_i)) = sum(dC/dV_j*dV_j/dV_i, j) = sum(dC/dV_j*w_ij, j)
#dC/db_i = dC/dV_i*dV_i/db_i = dC/dV_i
#implementation:
#dOut = dC/dV_o = dloss(out)/d(act(V_o))*dAct(V_o)
#dW_out = outer(dOut, act(values[, m]))
#could be a diff act function than the hidden layers
#vectorized:
#let dV = matrix(nrow n, ncol = m)
#dV[, i] = t(nn_weights[i, , ]) %*% dV[, i+1] * dact(values[, i])
#dV[, m] = out_weights %*% dOut
#these become the bias values
#let dW = array(dim = c(m, n, n))#hidden layers
#dW = outer product of dV_i and act(values_i)
#dW[i, , ] = outer(dV[, i+1], act(values[, i]))
#dW_in = outer(dV[, 1], input)

#could run in parallel with run_net if written for just in time

run_backprop <- function(args) {
    output <- args$output
    correct_output <- args$correct_output
    values <- args$values
    model <- args$model
    dloss <- args$dloss
    input <- args$input

    out_weights <- model$weights$out
    nn_weights <- model$weights$hidden
    act <- model$act$hidden
    dact <- model$dact$hidden
    n <- model$dim["n"]
    m <- model$dim["m"]

    dOut <- dloss(output - correct_output) # there is no dAct in the network
    dW_out <- outer(dOut, act(values[, m]))
    dW_out <- matrix(dW_out, nrow = model$dim["o"], ncol = model$dim["n"])
    dV <- matrix(0, nrow = n, ncol = m) #acts as dBias
    dV[, m] <- as.vector(t(out_weights) %*% dOut) * dact(values[, m])

    dW <- array(dim = c(m - 1, n, n))
    for (i in (m - 1):1) {
        dV[, i] <- as.vector(t(nn_weights[i, , ]) %*% dV[, i + 1]) *
            dact(values[, i])
        dW[i, , ] <- outer(dV[, i + 1], act(values[, i]))
    }
    dW_in <- outer(dV[, 1], input)
    list(dHidden = dW, dBOut = dOut, dBias = dV, dIn = dW_in, dOut = dW_out)
}
run_backprop(backprop_args)

#verify backprop:
#too lazy to do this for the whole thing, u get the idea
verify_backprop <- function(args) {
    backprop_vals <- run_backprop(args)
    input <- args$input
    model <- args$model
    e <- 1e-5
    err <- c()
    for (i in seq_along(model$biases$out)) {
        cp <- model
        cp$biases$out[[i]] <- model$biases$out[[i]] + e
        c1 <- loss(run_net(input, cp)$out - args$correct_output)

        cp$biases$out[[i]] <- model$biases$out[[i]] - e
        c2 <- loss(run_net(input, cp)$out - args$correct_output)

        num_grad <- (c1 - c2) / (2 * e)
        er <- backprop_vals$dBOut - num_grad
        err <- c(err, er)
    }
    err
}
verify_backprop(backprop_args)

#we could generalize by making one massive weight vector
#and then handle weight calculation internally
#thus allowing custom model architecture
get_grads <- function(model, inputs, outputs, num_grads = 1) {
    init <- FALSE
    grads <- NA
    for (i in seq_len(min(length(input), num_grads))) {
        output <- run_net(inputs[i], model)
        backprop_args <- list(
            output = output$output,
            correct_output = correct_output,
            values = output$values,
            model = model,
            dloss = dloss,
            input = inputs[i]
        )
        run_grads <- run_backprop(backprop_args)
        if (!init) {
            init <- TRUE
            grads <- run_grads
            for (j in seq_along(grads)) {
                grads[[j]] <- grads[[j]] / length(inputs)
            }
        } else {
            for (j in seq_along(run_grads)) {
                grads[[j]] <- grads[[j]] + run_grads[[j]] / length(inputs)
            }
        }
    }
    grads
}
#grad_desc is model specific(e.g. we cannot run LSTM or other architectures without modification)
grad_desc <- function(
    model,
    inputs,
    outputs,
    times = 1,
    optimizer = "ADAM",
    method = "Stochastic"
) {
    #R is copy on modify, so we can just make a copy
    trainer <- model #we don't want to modify original array, this just makes it clearer

    step_size <- list(weights = model$weights, biases = model$biases)
    #step size decreases exponentially with ADAM, stays constant with other optimizers
    #one step for each weight/bias
    for (i in names(step_size$weights)) {
        #set all weights to 1
        step_size$weights[[i]] <- array(1, dim(step_size$weights[[i]]))
    }
    for (i in names(step_size$biases)) {
        #can generalize; won't.
        step_size$biases[[i]] <- array(1, dim(step_size$biases[[i]]))
    }
    for (i in seq_len(times)) {}
}
grad_desc(model, c(1, 2, 3), c(3, 2, 1))
