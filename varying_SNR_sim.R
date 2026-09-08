rm(list = ls())

library(ranger)
library(xtable)
library(progress)

set.seed(7595955)

# options
n <- 500L # training data size 
n_test <- 3*n # test data size 
p <- 5L
betas <- c(0.0, 0.5, 3.0)
mns.vec <- c(5, 10, 20, 50, 100, 200, 300, 400, 500) # node sizes for RF
num.trees <- 750L # 500 is the default
RFtrim.mns <- 3L # 5 is the default
alpha.vec <- seq(0, 3.0, 0.1)
alpha.cc.vec <- seq(0, 60, 2)
n_mods <- length(mns.vec) + 4 # alphatrim, CC-trim, RFs, tuned RF, tree


# generate training and test data
dat_list <- list()
dat_test_list <- list()
for (i in 1:length(betas)) { # varying SNR data
  beta <- betas[i]
  x <- matrix(runif(n*p, 0.0, 1.0), nrow = n, ncol = p)
  y <- apply(x, MARGIN = 1, FUN = function(z) rnorm(n = 1, mean = sum(beta * z), sd = 1.0))
  dat <- as.data.frame(cbind(y, x))
  dat_list[[i]] <- dat
  x_test <- matrix(runif(n_test*p, 0.0, 1.0), nrow = n_test, ncol = p)
  y_test <- apply(x_test, MARGIN = 1, FUN = function(z) rnorm(n = 1, mean = sum(beta * z), sd = 1.0))
  dat_test <- as.data.frame(cbind(y_test, x_test))
  dat_test_list[[i]] <- dat_test
}
rm(i)

# mixed SNR (elbow) data
x <- matrix(runif(n*p, 0.0, 1.0), nrow = n, ncol = p)
y <- apply(x, MARGIN = 1, 
           FUN = function(z) rnorm(n = 1, mean = 10 * (z[1] - 0.5) * as.numeric(z[1] >= 0.5), sd = 1.0))
dat <- as.data.frame(cbind(y, x))
dat_list[[length(dat_list) + 1]] <- dat
x_test <- matrix(runif(n_test*p, 0.0, 1.0), nrow = n_test, ncol = p)
y_test <- apply(x_test, MARGIN = 1, 
           FUN = function(z) rnorm(n = 1, mean = 10 * (z[1] - 0.5) * as.numeric(z[1] >= 0.5), sd = 1.0))
dat_test <- as.data.frame(cbind(y_test, x_test))
dat_test_list[[length(dat_test_list) + 1]] <- dat_test


# prepare simulation matrices
sim_data <- matrix(NA, nrow = n_mods, ncol = length(dat_list)) # #algos x #data sets
rownames(sim_data) <- c("AlphaTrim", "CC-Trim", paste0("RF", mns.vec), "Tuned RF", "Tree")

#### Simulation ----------------------------------------------------------------

Train.Models <- function(dat, mns.vec, num.trees, RFtrim.mns, alpha.vec, alpha.cc.vec) {
  mods <- vector(mode = 'list', length = n_mods)

  # alpha-trimming (BIC and CC share the same forest)
  mod <- ranger(
    y ~ ., data=dat, write.forest = TRUE, min.node.size = RFtrim.mns,
    num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE,
    max.depth = NULL, mtry = NULL
  )
  mods[[1]] <- alpha.trim.tune(mod, ycol = 1, criterion = "BIC", verbose = TRUE, alpha = alpha.vec)
  mods[[2]] <- alpha.trim.tune(mod, ycol = 1, criterion = "CC", verbose = TRUE, alpha = alpha.cc.vec)

  # RFs
  i <- 3
  tuned_rf <- NULL
  best_mspe <- .Machine$double.xmax
  for (mns in mns.vec) {
    mods[[i]] <- ranger(
      y ~ ., data=dat, write.forest = TRUE, min.node.size = mns, num.trees = num.trees,
      oob.error = TRUE, keep.inbag = TRUE, max.depth = NULL, mtry = NULL
    )
    if (mods[[i]]$prediction.error < best_mspe) {
      best_mspe <- mods[[i]]$prediction.error 
      tuned_rf <- mods[[i]]
    }
    i <- i+1
  }
  
  # tuned RF 
  mods[[i]] <- tuned_rf # best tree based on OOB MSPE
  
  return(mods)
}


RMSPE <- function(preds, dat_test) {
  return(sqrt(mean((preds - dat_test$y)^2)))
}


Test.Metrics <- function(mods, dat_test) {
  metrics <- numeric(length(mods))
  
  # alpha-trimming (BIC)
  preds <- predict(mods[[1]]$object.trim.min, data = dat_test)
  metrics[1] <- RMSPE(preds, dat_test)

  # CC-trimming
  preds <- predict(mods[[2]]$object.trim.min, data = dat_test)
  metrics[2] <- RMSPE(preds, dat_test)

  # RFs and tuned RF
  for (i in 3:(length(metrics)-1)) {
    preds <- predict(mods[[i]], data = dat_test)$predictions
    metrics[i] <- RMSPE(preds, dat_test)
  }
  
  return(metrics)
}


# run simulation
for (j in 1:ncol(sim_data)) { # loop through data
  dat <- dat_list[[j]]
  dat_test <- dat_test_list[[j]]
  mods <- Train.Models(dat, mns.vec, num.trees, RFtrim.mns, alpha.vec, alpha.cc.vec) # train models
  metrics <- Test.Metrics(mods, dat_test) # collect RMSPE on test set
  sim_data[, j] <- metrics
}
sim_data

xtable(sim_data, digits = 3)


