#!/usr/bin/env Rscript
# =============================================================================
# zone_scores.R — zone-average verification scores with the `verification` package
# =============================================================================
# Called by eccas_s2s.validate.r_bridge. Reads one CSV of forecast/observation
# pairs (one row per year and per period) and writes the score CSVs next to it.
#
# Input columns (see eccas_s2s.validate.pairs):
#   period, year, ensmean, ens_sd, obs, pBN, pNN, pAN, obs_cat
#
# Method taken from the CAPC-AC reference chain (compute_scores_v2.R), with two
# deliberate differences:
#   * brier() is called with fine thresholds: its default bins the probabilities
#     into ten classes, which shifts the score. Its own reliability/resolution
#     decomposition is then unusable (with one forecast per class, reliability
#     collapses onto the BS and resolution onto the uncertainty), and the
#     package's binned decomposition does not close either (rel - res + unc
#     differs from its own bs). The decomposition is therefore computed here,
#     over the same ten classes as the reliability diagram, so that
#     rel - res + unc = bs_binned exactly (checked by a test);
#   * every field of the package is read through getf(), which returns NA when a
#     field is missing instead of aborting the block.
#
# Usage: Rscript zone_scores.R <pairs.csv> <out_dir> <label>
# =============================================================================

suppressPackageStartupMessages(library(verification))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) { cat("Usage: Rscript zone_scores.R <pairs.csv> <out_dir> <label>\n"); quit(status = 1) }
PAIRS <- args[1]; OUT <- args[2]; LABEL <- args[3]
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

N_CAT <- 3; CLIM_P <- 1 / N_CAT; EPS <- 1e-6; N_BINS <- 10
FINE <- seq(0, 1, 1e-4)      # fine thresholds: see the note above

getf <- function(obj, name, digits = 6) {
  if (is.null(obj)) return(NA_real_)
  v <- tryCatch(obj[[name]], error = function(e) NULL)
  if (is.null(v) || length(v) == 0 || !is.numeric(v)) return(NA_real_)
  round(as.numeric(v[1]), digits)
}

ignorance_score <- function(p) { p <- pmax(pmin(p, 1 - EPS), EPS); -mean(log2(p), na.rm = TRUE) }
ignorance_skill <- function(ign) 1 - ign / (-log2(CLIM_P))

# GROC / 2-AFC for ordered categories (Mason & Weigel 2009)
groc_score <- function(P, obs_cat) {
  n <- length(obs_cat); if (n < 2) return(NA_real_)
  num <- 0; den <- 0
  for (i in seq_len(n - 1)) for (j in (i + 1):n) {
    if (obs_cat[i] == obs_cat[j]) next
    den <- den + 1
    if (obs_cat[i] < obs_cat[j]) { pk <- P[i, ]; pl <- P[j, ] } else { pk <- P[j, ]; pl <- P[i, ] }
    m <- length(pk); numer <- 0
    for (s in seq_len(m - 1)) numer <- numer + pk[s] * sum(pl[(s + 1):m])
    denom <- 1 - sum(pk * pl)
    f <- if (denom <= 0) 0.5 else numer / denom
    num <- num + if (f > 0.5) 1 else if (f == 0.5) 0.5 else 0
  }
  if (den == 0) NA_real_ else num / den
}

# Murphy decomposition over the reliability classes:
#   BS(classes) = fiabilite - resolution + incertitude
# with fiabilite = sum n_k (f_k - o_k)^2 / N, resolution = sum n_k (o_k - obar)^2 / N
# and incertitude = obar (1 - obar).
brier_decomposition <- function(prob, event) {
  bks <- seq(0, 1, length.out = N_BINS + 1)
  n <- length(event); obar <- mean(event)
  rel <- 0; res <- 0; bs_bin <- 0
  for (ib in seq_len(N_BINS)) {
    inb <- prob >= bks[ib] & (if (ib == N_BINS) prob <= bks[ib + 1] else prob < bks[ib + 1])
    inb[is.na(inb)] <- FALSE
    nk <- sum(inb)
    if (nk == 0) next
    fk <- mean(prob[inb]); ok <- mean(event[inb])
    rel <- rel + nk * (fk - ok)^2
    res <- res + nk * (ok - obar)^2
    bs_bin <- bs_bin + nk * ((fk - ok)^2 - (ok - obar)^2)
  }
  unc <- obar * (1 - obar)
  list(reliability = rel / n, resolution = res / n, uncertainty = unc,
       bs_binned = bs_bin / n + unc)
}

reliability_bins <- function(prob, event, period, item) {
  bks <- seq(0, 1, length.out = N_BINS + 1)
  ctr <- (bks[-length(bks)] + bks[-1]) / 2
  do.call(rbind, lapply(seq_len(N_BINS), function(ib) {
    inb <- prob >= bks[ib] & (if (ib == N_BINS) prob <= bks[ib + 1] else prob < bks[ib + 1])
    cnt <- sum(inb, na.rm = TRUE)
    data.frame(label = LABEL, period = period, item = item, bin_center = ctr[ib],
               mean_forecast = if (cnt > 0) round(mean(prob[inb], na.rm = TRUE), 6) else NA,
               obs_frequency = if (cnt > 0) round(mean(event[inb], na.rm = TRUE), 6) else NA,
               count = cnt, stringsAsFactors = FALSE)
  }))
}

pairs <- read.csv(PAIRS, stringsAsFactors = FALSE)
periods <- unique(pairs$period)
det_rows <- list(); ter_rows <- list(); cat_rows <- list(); rel_rows <- list()

for (per in periods) {
  d <- pairs[pairs$period == per, ]
  ok <- is.finite(d$ensmean) & is.finite(d$obs)
  d <- d[ok, ]
  n <- nrow(d)
  if (n < 3) next

  # ---------------------------------------------------------- deterministic
  err <- d$ensmean - d$obs
  mse <- mean(err^2); mse_clim <- mean((d$obs - mean(d$obs))^2)
  acc <- if (sd(d$ensmean) > 0 && sd(d$obs) > 0) cor(d$ensmean, d$obs) else NA_real_
  acc_p <- if (!is.na(acc) && abs(acc) < 1)
    2 * pt(-abs(acc * sqrt((n - 2) / (1 - acc^2))), df = n - 2) else NA_real_
  spear <- if (sd(d$ensmean) > 0 && sd(d$obs) > 0)
    suppressWarnings(cor(d$ensmean, d$obs, method = "spearman")) else NA_real_
  crps_v <- NA_real_; ign_g <- NA_real_
  if ("ens_sd" %in% names(d)) {
    good <- is.finite(d$ens_sd) & d$ens_sd > 0
    if (sum(good) >= 3) {
      cc <- tryCatch(crps(obs = d$obs[good], pred = cbind(d$ensmean[good], d$ens_sd[good])),
                     error = function(e) NULL)
      crps_v <- getf(cc, "CRPS"); ign_g <- getf(cc, "IGN")
    }
  }
  det_rows[[length(det_rows) + 1]] <- data.frame(
    label = LABEL, period = per, n_years = n,
    bias = round(mean(err), 6), mae = round(mean(abs(err)), 6), rmse = round(sqrt(mse), 6),
    rmse_clim = round(sqrt(mse_clim), 6),
    msess = round(if (mse_clim > 0) 1 - mse / mse_clim else NA_real_, 6),
    pearson = round(acc, 6), acc = round(acc, 6), acc_pvalue = round(acc_p, 6),
    spearman = round(spear, 6), crps = crps_v, ign_gaussian = ign_g,
    stringsAsFactors = FALSE)

  # ---------------------------------------------------------- terciles
  need <- c("pBN", "pNN", "pAN", "obs_cat")
  if (!all(need %in% names(d))) next
  dd <- d[complete.cases(d[, need]), ]
  if (nrow(dd) < 3) next
  P <- as.matrix(dd[, c("pBN", "pNN", "pAN")])
  rs <- rowSums(P); P <- P / ifelse(rs == 0, 1, rs)
  obs1 <- as.integer(dd$obs_cat) + 1L

  r <- tryCatch(rps(obs = obs1, pred = P, baseline = rep(CLIM_P, N_CAT)), error = function(e) NULL)
  p_ver <- P[cbind(seq_len(nrow(P)), obs1)]
  ign <- ignorance_score(p_ver)
  groc <- tryCatch(groc_score(P, dd$obs_cat), error = function(e) NA_real_)
  fcst_cat <- apply(P, 1, which.max)
  cm <- table(factor(fcst_cat, levels = 1:3), factor(obs1, levels = 1:3))
  mc <- tryCatch(multi.cont(as.matrix(cm)), error = function(e) NULL)
  ter_rows[[length(ter_rows) + 1]] <- data.frame(
    label = LABEL, period = per, n_years = nrow(dd),
    rps = getf(r, "rps"), rps_clim = getf(r, "rps.clim"), rpss = getf(r, "rpss"),
    groc = round(groc, 6), ignorance = round(ign, 6), ign_skill = round(ignorance_skill(ign), 6),
    hss = getf(mc, "hss"), pss = getf(mc, "pss"), gerrity = getf(mc, "gs"), pc = getf(mc, "pc"),
    stringsAsFactors = FALSE)

  # ------------------------------------------------- one row per category
  for (k in seq_len(N_CAT)) {
    name <- c("BN", "NN", "AN")[k]
    ev <- as.integer(dd$obs_cat == (k - 1))
    pr <- P[, k]
    # exact score from the package (fine thresholds), decomposition computed over
    # the reliability classes: see the note at the top of the file.
    b <- tryCatch(brier(obs = ev, pred = pr, baseline = rep(CLIM_P, length(ev)), thresholds = FINE),
                  error = function(e) NULL)
    dc <- brier_decomposition(pr, ev)
    ra <- if (length(unique(ev)) > 1)
      tryCatch(roc.area(obs = ev, pred = pr), error = function(e) NULL) else NULL
    cat_rows[[length(cat_rows) + 1]] <- data.frame(
      label = LABEL, period = per, category = name, n_years = length(ev),
      base_rate = round(mean(ev), 6), bs = getf(b, "bs"), bs_clim = getf(b, "bs.baseline"),
      bss = getf(b, "ss"), bs_binned = round(dc$bs_binned, 6),
      bs_reliability = round(dc$reliability, 6), bs_resolution = round(dc$resolution, 6),
      bs_uncertainty = round(dc$uncertainty, 6), n_bins_decomposition = N_BINS,
      roc_area = getf(ra, "A"), roc_pvalue = getf(ra, "p.value"),
      stringsAsFactors = FALSE)
    rel_rows[[length(rel_rows) + 1]] <- reliability_bins(pr, ev, per, name)
  }
}

write_rows <- function(rows, name) {
  if (length(rows) == 0) return(invisible(NULL))
  write.csv(do.call(rbind, rows), file.path(OUT, name), row.names = FALSE)
}
write_rows(det_rows, "deterministic_scores.csv")
write_rows(ter_rows, "tercile_scores.csv")
write_rows(cat_rows, "category_scores.csv")
write_rows(rel_rows, "reliability_bins.csv")
cat(sprintf("zone_scores.R: %s — %d période(s)\n", LABEL, length(periods)))
