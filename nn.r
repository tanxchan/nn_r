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

act <- function (input){#input is a vector(doesn't rly matter dimension)
    return (pmax(0, input))# relu
}
dact <- function(input){
    as.integer(input > 0)
}
loss <- function(input){#loss function, usually mean squared or smth
    #using mean squared bc the total loss is j the derivative is nice
    sum(input ^ 2)
}
dloss <- function(input){
    2 * input
}

values <- matrix(nrow = n, ncol = m)

run_net <- function(input){
    values[, 1] <- in_weights %*% input + biases[, 1]
    for (i in 2:m){
        values[, i] <- nn_weights[i - 1, , ] %*%
        act(values[, i - 1]) + biases[, i]
    }
    out <- out_weights %*% act(values[, m]) + out_biases#could make this sigmoid function
    return (out)
}

input <- c(3)
output <- run_net(input)
correct_output <- c(1)

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
run_backprop <- function(){
    dOut <- dloss(output - correct_output) # there is no dAct in the network
    dW_out <- outer(dOut, act(values[, m]))
    dV <- matrix(0, nrow = n, ncol = m)
    dV[, m] <- as.vector(t(out_weights) %*% dOut) * dact(values[, m])

    dW <- array(dim = c(m - 1, n, n))
    for (i in (m - 1):1){
        dV[, i] <- as.vector(t(nn_weights[i, , ]) %*% dV[, i + 1]) * dact(values[, i])
        dW[i, , ] <- outer(dV[, i + 1], act(values[, i]))
    }
    dW_in <- outer(dV[, 1], input)
}