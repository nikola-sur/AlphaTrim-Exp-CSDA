rm(list = ls())

library(ranger)

set.seed(7595955)

# options
n <- 500L
n_test <- 3*n
p <- 5L
betas <- c(0.0, 0.5, 3.0)
mns.vec <- c(5, 10, 20, 50, 100, 200, 300, 400, 500)
num.trees <- 750L
RFtrim.mns <- 3L
alpha.vec <- seq(0, 3.0, 0.1)
alpha.cc.vec <- seq(0, 60, 2)
n_reps <- 20L
n_dgps <- length(betas) + 1 # 3 varying-SNR + 1 elbow
mod_names <- c("AlphaTrim", "CC-Trim", paste0("RF", mns.vec), "Tuned RF")
n_mods <- length(mod_names)

# DGP generators
generate_data <- function(n, p, dgp_idx, betas) {
  x <- matrix(runif(n * p), nrow = n, ncol = p)
  if (dgp_idx <= length(betas)) {
    beta <- betas[dgp_idx]
    y <- apply(x, 1, function(z) rnorm(1, mean = sum(beta * z), sd = 1.0))
  } else {
    y <- apply(x, 1, function(z) rnorm(1, mean = 10 * (z[1] - 0.5) * as.numeric(z[1] >= 0.5), sd = 1.0))
  }
  as.data.frame(cbind(y, x))
}

RMSPE <- function(preds, dat_test) {
  sqrt(mean((preds - dat_test$y)^2))
}

# 3D array: methods x DGPs x replicates
results <- array(NA, dim = c(n_mods, n_dgps, n_reps),
                 dimnames = list(mod_names, c(paste0("beta=", betas), "Elbow"), NULL))

cat(sprintf("Starting %d replicates x %d DGPs x %d methods\n", n_reps, n_dgps, n_mods))

for (rep in 1:n_reps) {
  t0 <- proc.time()
  for (dgp in 1:n_dgps) {
    dat <- generate_data(n, p, dgp, betas)
    dat_test <- generate_data(n_test, p, dgp, betas)

    # Grow one forest for both BIC and CC trimming
    mod <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = RFtrim.mns,
                  num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE)
    bic_res <- alpha.trim.tune(mod, ycol = 1, criterion = "BIC", verbose = FALSE, alpha = alpha.vec)
    cc_res  <- alpha.trim.tune(mod, ycol = 1, criterion = "CC",  verbose = FALSE, alpha = alpha.cc.vec)

    results["AlphaTrim", dgp, rep] <- RMSPE(predict(bic_res$object.trim.min, data = dat_test), dat_test)
    results["CC-Trim",   dgp, rep] <- RMSPE(predict(cc_res$object.trim.min,  data = dat_test), dat_test)

    # RFs with various min.node.size
    best_mspe <- .Machine$double.xmax
    best_rf <- NULL
    for (mns in mns.vec) {
      rf <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = mns,
                   num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE)
      rf_name <- paste0("RF", mns)
      results[rf_name, dgp, rep] <- RMSPE(predict(rf, data = dat_test)$predictions, dat_test)
      if (rf$prediction.error < best_mspe) {
        best_mspe <- rf$prediction.error
        best_rf <- rf
      }
    }
    results["Tuned RF", dgp, rep] <- RMSPE(predict(best_rf, data = dat_test)$predictions, dat_test)
  }
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Rep %2d/%d done (%.1f sec)\n", rep, n_reps, elapsed))
}

# Compute mean and SE
sim_mean <- apply(results, c(1, 2), mean)
sim_se   <- apply(results, c(1, 2), function(x) sd(x) / sqrt(length(x)))

cat("\n=== Mean RMSPE ===\n")
print(round(sim_mean, 4))
cat("\n=== SE ===\n")
print(round(sim_se, 4))

# Save results
saveRDS(list(results = results, mean = sim_mean, se = sim_se),
        file = "varying_SNR_20rep_results.rds")
cat("\nResults saved to varying_SNR_20rep_results.rds\n")
