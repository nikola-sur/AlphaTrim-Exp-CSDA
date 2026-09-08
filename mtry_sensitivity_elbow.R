rm(list = ls())

library(ranger)

set.seed(8492013)

# options
n <- 500L
n_test <- 3*n
p <- 5L
num.trees <- 750L
RFtrim.mns <- 3L
alpha.vec <- seq(0, 3.0, 0.1)
alpha.cc.vec <- seq(0, 60, 2)
mns.vec <- c(5, 10, 20, 50, 100, 200, 300, 400, 500)
mtry.vec <- 1:p

mod_names <- c("AlphaTrim", "CC-Trim", paste0("RF", mns.vec), "Tuned RF")
n_mods <- length(mod_names)

RMSPE <- function(preds, dat_test) {
  sqrt(mean((preds - dat_test$y)^2))
}

# Elbow DGP generator
generate_elbow <- function(n, p) {
  x <- matrix(runif(n * p), nrow = n)
  y <- apply(x, 1, function(z) rnorm(1, mean = 10 * (z[1] - 0.5) * as.numeric(z[1] >= 0.5), sd = 1.0))
  as.data.frame(cbind(y, x))
}

# Results: methods x mtry
results <- matrix(NA, nrow = n_mods, ncol = length(mtry.vec),
                  dimnames = list(mod_names, paste0("mtry=", mtry.vec)))

dat <- generate_elbow(n, p)
dat_test <- generate_elbow(n_test, p)

for (m_idx in seq_along(mtry.vec)) {
  m <- mtry.vec[m_idx]
  cat(sprintf("mtry = %d ...\n", m))

  # BIC and CC alpha-trimming (shared forest)
  mod <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = RFtrim.mns,
                num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE, mtry = m)
  bic_res <- alpha.trim.tune(mod, ycol = 1, criterion = "BIC", verbose = FALSE, alpha = alpha.vec)
  cc_res  <- alpha.trim.tune(mod, ycol = 1, criterion = "CC",  verbose = FALSE, alpha = alpha.cc.vec)

  results["AlphaTrim", m_idx] <- RMSPE(predict(bic_res$object.trim.min, data = dat_test), dat_test)
  results["CC-Trim",   m_idx] <- RMSPE(predict(cc_res$object.trim.min,  data = dat_test), dat_test)

  # RFs with various min.node.size
  best_mspe <- .Machine$double.xmax
  best_rf <- NULL
  for (mns in mns.vec) {
    rf <- ranger(y ~ ., data = dat, write.forest = TRUE, min.node.size = mns,
                 num.trees = num.trees, oob.error = TRUE, keep.inbag = TRUE, mtry = m)
    rf_name <- paste0("RF", mns)
    results[rf_name, m_idx] <- RMSPE(predict(rf, data = dat_test)$predictions, dat_test)
    if (rf$prediction.error < best_mspe) {
      best_mspe <- rf$prediction.error
      best_rf <- rf
    }
  }
  results["Tuned RF", m_idx] <- RMSPE(predict(best_rf, data = dat_test)$predictions, dat_test)
}

cat("\n=== RMSPE: Elbow DGP across mtry values ===\n")
print(round(results, 4))

saveRDS(results, file = "mtry_sensitivity_elbow_results.rds")
cat("\nResults saved to mtry_sensitivity_elbow_results.rds\n")
