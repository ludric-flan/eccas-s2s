#!/usr/bin/env Rscript
# =============================================================================
# zone_diagrams.R — reliability (attributes) and ROC diagrams, in R
# =============================================================================
# Called by eccas_s2s.validate.r_bridge.run_zone_diagrams. Reads the pooled
# grid-point pairs of one zone (eccas_s2s.validate.pooled) and draws, per
# period, two separate figures, each carrying the three tercile categories on
# the same axes so they can be compared directly:
#
#   reliability_<period>.png  attributes diagram (reliability curves + no-skill
#          line + skill region + sharpness histogram), 95 % block-bootstrap
#          interval on each bin;
#   roc_<period>.png          ROC curves with their bootstrap envelopes and the
#          areas under the curve.
#
# Style and method taken from the reference chain of the CAPC-AC
# (plot_scores_v2.R, plot_pooled_diagrams_v2.R): base R graphics only, under-
# populated bins drawn as grey crosses outside the line, years resampled whole
# so the interval respects the dependence between neighbouring grid points.
#
# The areas come from verification::roc.area, so the figure and the CSV scores
# are computed by the same implementation (Draft §5.2).
#
# Files are written as <out_dir>/reliability/<period>.png and
# <out_dir>/roc/<period>.png, one metric per folder, as the rest of the chain does.
#
# Input columns : period (key, used for the file name), period_label (French label
# for the title), year, cell, obs_cat, pBN, pNN, pAN
# Usage: Rscript zone_diagrams.R <pooled_pairs.csv> <out_dir> <label> [n_boot]
# =============================================================================

suppressPackageStartupMessages(library(verification))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript zone_diagrams.R <pooled_pairs.csv> <out_dir> <label> [n_boot]\n")
  quit(status = 1)
}
PAIRS <- args[1]; OUT <- args[2]; LABEL <- args[3]
N_BOOT <- if (length(args) > 3) as.integer(args[4]) else 200
dir.create(file.path(OUT, "reliability"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT, "roc"), recursive = TRUE, showWarnings = FALSE)

CATS      <- c("BN", "NN", "AN")
CAT_NAME  <- c(BN = "Below Normal", NN = "Near Normal", AN = "Above Normal")
CAT_COL   <- c(BN = "#E07B39", NN = "#4CAF50", AN = "#2196F3")
SPARSE_COL <- "grey55"
CLIM_P    <- 1 / 3
N_BINS    <- 10
MIN_PAIRS <- 20      # a bin below this is an anecdote
MIN_YEARS <- 5       # ... and so is a bin drawing on very few distinct years
PNG_RES   <- 130
set.seed(20260901)

png_open <- function(path, w = 1500, h = 1000) {
  ok <- tryCatch({ png(path, width = w, height = h, res = PNG_RES, type = "cairo"); TRUE },
                 error = function(e) FALSE)
  if (!ok) png(path, width = w, height = h, res = PNG_RES)
}

# ------------------------------------------------------------------ binning
bin_table <- function(prob, event, year) {
  bks <- seq(0, 1, length.out = N_BINS + 1)
  ctr <- (bks[-length(bks)] + bks[-1]) / 2
  do.call(rbind, lapply(seq_len(N_BINS), function(ib) {
    inb <- prob >= bks[ib] & (if (ib == N_BINS) prob <= bks[ib + 1] else prob < bks[ib + 1])
    inb[is.na(inb)] <- FALSE
    n <- sum(inb); ny <- length(unique(year[inb]))
    data.frame(bin_center = ctr[ib], n_pairs = n, n_years = ny,
               mean_frcst = if (n > 0) mean(prob[inb]) else NA_real_,
               obs_freq   = if (n > 0) mean(event[inb]) else NA_real_,
               sparse = as.integer(n < MIN_PAIRS || ny < MIN_YEARS))
  }))
}

# 95 % interval of each bin's observed frequency, years resampled whole
# (the grid points of a year travel together: they are not independent).
bin_ci <- function(prob, event, year, tab) {
  ys <- unique(year); ny <- length(ys)
  if (ny < 4 || N_BOOT < 20) return(cbind(ci_lo = NA_real_, ci_hi = NA_real_))
  idx <- split(seq_along(year), as.character(year))
  boot <- matrix(NA_real_, nrow = N_BOOT, ncol = nrow(tab))
  for (b in seq_len(N_BOOT)) {
    take <- unlist(idx[as.character(sample(ys, ny, replace = TRUE))], use.names = FALSE)
    boot[b, ] <- bin_table(prob[take], event[take], year[take])$obs_freq
  }
  q <- t(apply(boot, 2, function(v) {
    v <- v[is.finite(v)]
    if (length(v) < 10) c(NA_real_, NA_real_) else unname(quantile(v, c(0.025, 0.975)))
  }))
  colnames(q) <- c("ci_lo", "ci_hi")
  q
}

# --------------------------------------------------------------------- ROC
roc_points <- function(prob, event) {
  # empirical ROC: every distinct probability is a decision threshold, the
  # first one (Inf) giving the origin "never forecast the event".
  thr <- c(Inf, sort(unique(prob), decreasing = TRUE))
  hr <- sapply(thr, function(t) mean(prob[event == 1] >= t))
  far <- sapply(thr, function(t) mean(prob[event == 0] >= t))
  data.frame(threshold = thr, HR = hr, FAR = far)
}

# Mann-Whitney form of the area under the ROC curve, identical to the A of
# verification::roc.area (checked in the tests) but usable inside the bootstrap,
# where roc.area's p-value machinery fails on a resample with repeated rows.
auc_rank <- function(prob, event) {
  n1 <- sum(event == 1); n0 <- sum(event == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(prob)
  (sum(r[event == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

roc_envelope <- function(prob, event, year, grid = seq(0, 1, 0.02)) {
  ys <- unique(year); ny <- length(ys)
  if (ny < 4 || N_BOOT < 20) return(NULL)
  idx <- split(seq_along(year), as.character(year))
  curves <- matrix(NA_real_, nrow = N_BOOT, ncol = length(grid))
  aucs <- rep(NA_real_, N_BOOT)
  for (b in seq_len(N_BOOT)) {
    take <- unlist(idx[as.character(sample(ys, ny, replace = TRUE))], use.names = FALSE)
    e <- event[take]; p <- prob[take]
    if (length(unique(e)) < 2) next
    r <- roc_points(p, e)
    curves[b, ] <- approx(r$FAR, r$HR, xout = grid, ties = max, rule = 2)$y
    aucs[b] <- auc_rank(p, e)
  }
  list(grid = grid,
       lo = apply(curves, 2, quantile, 0.025, na.rm = TRUE),
       hi = apply(curves, 2, quantile, 0.975, na.rm = TRUE),
       auc_ci = quantile(aucs, c(0.025, 0.975), na.rm = TRUE))
}

# ------------------------------------------------------------------ panels
# Sharpness in its own panel under the diagram: drawn as an inset it covered the
# markers and the confidence bars of the upper-right bins.
sharp_panel <- function(tabs) {
  cmax <- max(unlist(lapply(tabs, function(t) max(t$n_pairs, na.rm = TRUE))), na.rm = TRUE)
  plot(NA, xlim = c(0, 1), ylim = c(0, cmax * 1.12), xaxs = "i", yaxs = "i",
       xlab = "Probabilité prévue", ylab = "Effectif", cex.lab = 1.0, cex.axis = 0.9,
       main = sprintf("Netteté — %d couples par catégorie (%d classes)",
                      sum(tabs[[1]]$n_pairs, na.rm = TRUE), nrow(tabs[[1]])),
       font.main = 1, cex.main = 0.85)
  grid(nx = NA, ny = NULL, col = "grey90")
  nb <- nrow(tabs[[1]]); bw <- 1 / nb; sub <- bw / (length(CATS) + 0.6)
  for (ib in seq_len(nb)) for (k in seq_along(CATS)) {
    t <- tabs[[k]]
    if (!is.finite(t$n_pairs[ib]) || t$n_pairs[ib] <= 0) next
    xa <- (ib - 1) * bw + (k - 1) * sub + sub * 0.35
    rect(xa, 0, xa + sub * 0.85, t$n_pairs[ib],
         col = if (t$sparse[ib] == 0) CAT_COL[CATS[k]] else SPARSE_COL, border = NA)
  }
  abline(h = 0, col = "grey40")
  box()
}

# The three categories share one set of axes (one figure per period), so the
# reader sees at a glance whether the overconfidence of "below normal" is the
# same as that of "above normal".
rel_figure <- function(tabs, main) {
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i", xlab = "Probabilité prévue",
       ylab = "Fréquence observée", main = main, font.main = 2, cex.main = 1.0,
       cex.lab = 1.0, cex.axis = 0.9)
  # skill region of the attributes diagram: a bin helps the Brier skill score
  # when it lies beyond the no-skill line (x + clim)/2, away from climatology.
  xs <- seq(0, 1, length.out = 200); ns <- (xs + CLIM_P) / 2
  rgt <- xs >= CLIM_P; lft <- xs <= CLIM_P
  polygon(c(xs[rgt], rev(xs[rgt])), c(ns[rgt], rep(1, sum(rgt))),
          col = adjustcolor("green", alpha.f = 0.08), border = NA)
  polygon(c(xs[lft], rev(xs[lft])), c(ns[lft], rep(0, sum(lft))),
          col = adjustcolor("green", alpha.f = 0.08), border = NA)
  lines(xs, ns, col = "grey55", lwd = 1)
  abline(h = CLIM_P, col = "grey65", lty = 3); abline(v = CLIM_P, col = "grey65", lty = 3)
  abline(0, 1, lty = 2, lwd = 1.3); grid(col = "grey90")

  cmax <- max(unlist(lapply(tabs, function(t) max(t$n_pairs, na.rm = TRUE))), na.rm = TRUE)
  for (k in seq_along(CATS)) {
    col <- CAT_COL[CATS[k]]
    s <- tabs[[k]]
    s <- s[is.finite(s$mean_frcst) & is.finite(s$obs_freq), ]
    if (!nrow(s)) next
    has <- is.finite(s$ci_lo) & is.finite(s$ci_hi)
    if (any(has))
      segments(s$mean_frcst[has], s$ci_lo[has], s$mean_frcst[has], s$ci_hi[has],
               col = adjustcolor(ifelse(s$sparse[has] == 0, col, SPARSE_COL), alpha.f = 0.75),
               lwd = 1.4)
    g <- s[s$sparse == 0, ]
    if (nrow(g) >= 2) lines(g$mean_frcst, g$obs_freq, col = col, lwd = 2.4)
    if (nrow(g)) points(g$mean_frcst, g$obs_freq, pch = 21, bg = col, col = "black",
                        cex = 0.7 + 1.6 * (g$n_pairs / cmax))
    sp <- s[s$sparse == 1, ]
    if (nrow(sp)) points(sp$mean_frcst, sp$obs_freq, pch = 4, col = SPARSE_COL, lwd = 2, cex = 1.0)
  }
  legend("topleft", bty = "n", cex = 0.78, lwd = 2.4, pch = 21, pt.cex = 1.1,
         col = CAT_COL[CATS], pt.bg = CAT_COL[CATS], legend = CAT_NAME[CATS])
  legend("left", bty = "n", cex = 0.66, inset = c(0, 0.16),
         legend = c("bin sous-peuplé", "IC 95 % (bloc-année)", "zone de BSS positif"),
         pch = c(4, NA, 22), lty = c(NA, 1, NA), lwd = c(2, 1.4, NA),
         col = c(SPARSE_COL, SPARSE_COL, "grey60"),
         pt.bg = c(NA, NA, adjustcolor("green", alpha.f = 0.18)))

}

roc_figure <- function(rocs, envs, areas, main) {
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), asp = 1, xlab = "Taux de fausses alertes",
       ylab = "Taux de détection", main = main, font.main = 2, cex.main = 1.0,
       cex.lab = 1.0, cex.axis = 0.9)
  abline(0, 1, lty = 2); grid(col = "grey90")
  for (k in seq_along(CATS)) {
    col <- CAT_COL[CATS[k]]
    env <- envs[[k]]
    if (!is.null(env)) {
      ok <- is.finite(env$lo) & is.finite(env$hi)
      polygon(c(env$grid[ok], rev(env$grid[ok])), c(env$lo[ok], rev(env$hi[ok])),
              col = adjustcolor(col, alpha.f = 0.14), border = NA)
    }
    r <- rocs[[k]]; o <- order(r$FAR, r$HR)
    lines(r$FAR[o], r$HR[o], col = col, lwd = 2.4)
  }
  leg <- sapply(seq_along(CATS), function(k) {
    a <- areas[[k]]
    if (is.finite(a$lo)) sprintf("%s : AUC %.3f [%.3f, %.3f]", CAT_NAME[CATS[k]], a$auc, a$lo, a$hi)
    else sprintf("%s : AUC %.3f", CAT_NAME[CATS[k]], a$auc)
  })
  legend("bottomright", bty = "n", cex = 0.78, lwd = 2.4, col = CAT_COL[CATS], legend = leg)
  legend("topleft", bty = "n", cex = 0.7,
         legend = sprintf("N = %d (%d ans)  •  enveloppe IC 95 %% bloc-année",
                          areas[[1]]$n, areas[[1]]$n_years))
}

# --------------------------------------------------------------------- main
pairs <- read.csv(PAIRS, stringsAsFactors = FALSE)
if (!all(c("pBN", "pNN", "pAN") %in% names(pairs))) {
  cat(sprintf("zone_diagrams.R: %s — pas de probabilités (moyenne d'ensemble) : aucun diagramme\n",
              LABEL))
  quit(status = 0)
}

summary_rows <- list(); bin_rows <- list()
for (per in unique(pairs$period)) {
  d <- pairs[pairs$period == per, ]
  d <- d[is.finite(d$obs_cat) & is.finite(d$pBN) & is.finite(d$pNN) & is.finite(d$pAN), ]
  if (nrow(d) < 50) next
  P <- as.matrix(d[, c("pBN", "pNN", "pAN")])
  rs <- rowSums(P); P <- P / ifelse(rs == 0, 1, rs)

  tabs <- list(); rocs <- list(); areas <- list(); envs <- list()
  for (k in seq_along(CATS)) {
    ev <- as.integer(d$obs_cat == (k - 1)); pr <- P[, k]
    tab <- bin_table(pr, ev, d$year)
    tab <- cbind(tab, bin_ci(pr, ev, d$year, tab))
    ra <- if (length(unique(ev)) > 1)
      tryCatch(roc.area(obs = ev, pred = pr), error = function(e) NULL) else NULL
    env <- roc_envelope(pr, ev, d$year)
    area <- list(auc = if (is.null(ra)) NA_real_ else ra$A,
                 p = if (is.null(ra)) NA_real_ else ra$p.value,
                 lo = if (is.null(env)) NA_real_ else unname(env$auc_ci[1]),
                 hi = if (is.null(env)) NA_real_ else unname(env$auc_ci[2]),
                 n = length(ev), n_years = length(unique(d$year)))
    tabs[[k]] <- tab; rocs[[k]] <- roc_points(pr, ev); areas[[k]] <- area; envs[[k]] <- env
      bin_rows[[length(bin_rows) + 1]] <- data.frame(label = LABEL, period = per,
                                                   category = CATS[k], tab)
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      label = LABEL, period = per, category = CATS[k], n_pairs = area$n,
      n_years = area$n_years, base_rate = round(mean(ev), 6),
      roc_area = round(area$auc, 6), roc_pvalue = round(area$p, 6),
      roc_lo = round(area$lo, 6), roc_hi = round(area$hi, 6),
      resolution = round(sum(tab$n_pairs * (tab$obs_freq - mean(ev))^2, na.rm = TRUE) / sum(tab$n_pairs), 6),
      reliability = round(sum(tab$n_pairs * (tab$mean_frcst - tab$obs_freq)^2, na.rm = TRUE) / sum(tab$n_pairs), 6),
      stringsAsFactors = FALSE)
  }
  per_label <- if ("period_label" %in% names(d)) as.character(d$period_label[1]) else per
  sub <- sprintf("%s — %s", LABEL, per_label)

  png_open(file.path(OUT, "reliability", sprintf("%s.png", per)), w = 1150, h = 1400)
  op <- par(oma = c(0, 0, 3.2, 0))
  layout(matrix(c(1, 2), nrow = 2), heights = c(3.1, 1))
  par(mar = c(4.2, 4.4, 1.2, 1.4))
  rel_figure(tabs, "")
  par(mar = c(4.2, 4.4, 2.2, 1.4))
  sharp_panel(tabs)
  mtext("Diagramme de fiabilité (attributs) — terciles", outer = TRUE, font = 2,
        cex = 1.0, line = 1.2)
  mtext(sub, outer = TRUE, cex = 0.85, line = 0.0)
  layout(1); par(op); dev.off()

  png_open(file.path(OUT, "roc", sprintf("%s.png", per)), w = 1100, h = 1100)
  op <- par(mar = c(4.2, 4.2, 4.0, 1.0))
  roc_figure(rocs, envs, areas, "Diagramme ROC — terciles")
  mtext(sub, side = 3, line = 0.4, cex = 0.85)
  par(op); dev.off()
}

if (length(summary_rows))
  write.csv(do.call(rbind, summary_rows), file.path(OUT, "diagram_scores.csv"), row.names = FALSE)
if (length(bin_rows))
  write.csv(do.call(rbind, bin_rows), file.path(OUT, "diagram_bins.csv"), row.names = FALSE)
cat(sprintf("zone_diagrams.R: %s — %d figure(s)\n", LABEL,
            length(list.files(OUT, pattern = "[.]png$", recursive = TRUE))))
