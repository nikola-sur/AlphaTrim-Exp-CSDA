rm(list = ls())

library(ranger)

set.seed(4820173)

# options
n <- 500L
n_test <- 5000L
num.trees <- 750L
RFtrim.mns <- 3L
alpha.vec <- seq(0, 3.0, 0.1)
mns.vec <- c(5, 10, 20, 50, 100, 200, 300, 400, 500)
p.vec <- c(5L, 25L, 50L)
metric_names <- c("overall", "noise", "signal")

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

# Results: methods x (p * 3 metrics)
# For each p: overall, noise (x1 < 0.5), signal (x1 >= 0.5 or x2 in [0.3, 0.7])
results <- matrix(NA, nrow = n_mods, ncol = length(p.vec) * length(metric_names),
                  dimnames = list(mod_names,
                                  paste0(rep(paste0("p=", p.vec), each = length(metric_names)),
                                         "_", metric_names)))

for (p_idx in seq_along(p.vec)) {
  p <- p.vec[p_idx]
  cat(sprintf("\n=== p = %d (mtry = %d) ===\n", p, floor(sqrt(p))))
  t0 <- proc.time()

  dat <- generate_multiscale(n, p)
  dat_test <- generate_multiscale(n_test, p)

  x1_test <- dat_test[, 2]
  x2_test <- dat_test[, 3]
  noise_idx  <- which(x1_test < 0.5 & (x2_test < 0.3 | x2_test > 0.7))
  signal_idx <- setdiff(1:n_test, noise_idx)

  cat(sprintf("  Region sizes: noise=%d, signal=%d\n",
              length(noise_idx), length(signal_idx)))

  store_metrics <- function(name, preds) {
    col_base <- (p_idx - 1) * length(metric_names)
    results[name, col_base + 1] <<- RMSPE(preds, dat_test$y)
    results[name, col_base + 2] <<- RMSPE(preds[noise_idx], dat_test$y[noise_idx])
    results[name, col_base + 3] <<- RMSPE(preds[signal_idx], dat_test$y[signal_idx])
  }

  # AlphaTrim (BIC)
  mod <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = RFtrim.mns,
                num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE)
  bic_res <- alpha.trim.tune(mod, ycol = 1, criterion = "BIC", verbose = FALSE, alpha = alpha.vec)
  preds <- predict(bic_res$object.trim.min, data = dat_test)
  store_metrics("AlphaTrim", preds)
  cat(sprintf("  AlphaTrim: alpha_min = %.1f\n", bic_res$alpha.min))

  # RFs with various min.node.size
  best_mspe <- .Machine$double.xmax
  best_rf <- NULL
  best_mns <- NA
  for (mns in mns.vec) {
    rf <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = mns,
                 num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE)
    preds_rf <- predict(rf, data = dat_test)$predictions
    store_metrics(paste0("RF", mns), preds_rf)
    if (rf$prediction.error < best_mspe) {
      best_mspe <- rf$prediction.error
      best_rf <- rf
      best_mns <- mns
    }
  }
  preds_tuned <- predict(best_rf, data = dat_test)$predictions
  store_metrics("Tuned RF", preds_tuned)
  cat(sprintf("  Tuned RF: best min.node.size = %d\n", best_mns))

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Done in %.1f sec\n", elapsed))
}

cat("\n\n=== Full results ===\n")
print(round(results, 4))

cat("\n=== Key methods summary ===\n")
key_methods <- c("AlphaTrim", "RF5", "RF10", "RF20", "RF50", "Tuned RF")
print(round(results[key_methods, ], 4))

saveRDS(results, file = "high_dim_multiscale_results.rds")
cat("\nResults saved to high_dim_multiscale_results.rds\n")
