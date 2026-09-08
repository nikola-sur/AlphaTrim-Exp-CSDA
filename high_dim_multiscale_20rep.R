rm(list = ls())

library(ranger)

set.seed(9182736)

# options
n <- 500L
n_test <- 5000L
num.trees <- 750L
RFtrim.mns <- 3L
alpha.vec <- seq(0, 3.0, 0.1)
mns.vec <- c(5, 10, 20, 50, 100, 200, 300, 400, 500)
p.vec <- c(5L, 25L, 50L)
metric_names <- c("overall", "noise", "signal")
n_reps <- 20L

mod_names <- c("AlphaTrim", paste0("RF", mns.vec), "Tuned RF")
n_mods <- length(mod_names)

RMSPE <- function(preds, y_true) {
  sqrt(mean((preds - y_true)^2))
}

generate_multiscale <- function(n, p) {
  x <- matrix(runif(n * p), nrow = n)
  y <- apply(x, 1, function(z) {
    coarse <- 10 * max(z[1] - 0.5, 0)
    fine <- 5 * (z[2] - 0.3) * (0.7 - z[2]) * as.numeric(z[2] >= 0.3 & z[2] <= 0.7)
    rnorm(1, mean = coarse + fine, sd = 1.0)
  })
  as.data.frame(cbind(y, x))
}

# 4D array: methods x p_values x metrics x reps
results <- array(NA, dim = c(n_mods, length(p.vec), length(metric_names), n_reps),
                 dimnames = list(mod_names, paste0("p=", p.vec), metric_names, NULL))

cat(sprintf("Starting %d reps x %d p-values x %d methods\n", n_reps, length(p.vec), n_mods))

for (rep in 1:n_reps) {
  t0 <- proc.time()

  for (p_idx in seq_along(p.vec)) {
    p <- p.vec[p_idx]

    dat <- generate_multiscale(n, p)
    dat_test <- generate_multiscale(n_test, p)

    x1_test <- dat_test[, 2]
    x2_test <- dat_test[, 3]
    noise_idx  <- which(x1_test < 0.5 & (x2_test < 0.3 | x2_test > 0.7))
    signal_idx <- setdiff(1:n_test, noise_idx)

    store <- function(name, preds) {
      results[name, p_idx, "overall", rep] <<- RMSPE(preds, dat_test$y)
      results[name, p_idx, "noise",   rep] <<- RMSPE(preds[noise_idx], dat_test$y[noise_idx])
      results[name, p_idx, "signal",  rep] <<- RMSPE(preds[signal_idx], dat_test$y[signal_idx])
    }

    # AlphaTrim (BIC)
    mod <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = RFtrim.mns,
                  num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE)
    bic_res <- alpha.trim.tune(mod, ycol = 1, criterion = "BIC", verbose = FALSE, alpha = alpha.vec)
    store("AlphaTrim", predict(bic_res$object.trim.min, data = dat_test))

    # RFs with various min.node.size
    best_mspe <- .Machine$double.xmax
    best_rf <- NULL
    for (mns in mns.vec) {
      rf <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = mns,
                   num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE)
      store(paste0("RF", mns), predict(rf, data = dat_test)$predictions)
      if (rf$prediction.error < best_mspe) {
        best_mspe <- rf$prediction.error
        best_rf <- rf
      }
    }
    store("Tuned RF", predict(best_rf, data = dat_test)$predictions)
  }

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Rep %2d/%d done (%.1f sec)\n", rep, n_reps, elapsed))
}

# Compute mean and SE across reps for each (method, p, metric)
sim_mean <- apply(results, c(1, 2, 3), mean)
sim_se   <- apply(results, c(1, 2, 3), function(x) sd(x) / sqrt(length(x)))

cat("\n=== Mean RMSPE (overall) ===\n")
print(round(sim_mean[, , "overall"], 4))
cat("\n=== SE (overall) ===\n")
print(round(sim_se[, , "overall"], 4))

cat("\n=== Mean RMSPE (noise region) ===\n")
print(round(sim_mean[, , "noise"], 4))
cat("\n=== SE (noise) ===\n")
print(round(sim_se[, , "noise"], 4))
cat("\n=== Mean RMSPE (signal region) ===\n")
print(round(sim_mean[, , "signal"], 4))
cat("\n=== SE (signal) ===\n")
print(round(sim_se[, , "signal"], 4))

# Key methods summary with SE
cat("\n=== Key methods: mean ± SE (overall) ===\n")
key <- c("AlphaTrim", "RF5", "RF10", "RF20", "RF50", "Tuned RF")
for (m in key) {
  vals <- sprintf("%.4f±%.4f", sim_mean[m, , "overall"], sim_se[m, , "overall"])
  cat(sprintf("  %-12s %s\n", m, paste(vals, collapse = "  ")))
}

saveRDS(list(results = results, mean = sim_mean, se = sim_se),
        file = "high_dim_multiscale_20rep_results.rds")
cat("\nResults saved to high_dim_multiscale_20rep_results.rds\n")
