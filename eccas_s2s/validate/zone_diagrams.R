#!/usr/bin/env Rscript
# =============================================================================
# zone_diagrams.R — reliability (attributes) and ROC diagrams, in R
# =============================================================================
# Called by eccas_s2s.validate.r_bridge.run_zone_diagrams. Reads the pooled
# grid-point pairs of one **family of probabilistic products** (terciles, SPI
# classes, percentile thresholds, millimetre thresholds) in long form and draws,
# per period, two separate figures, each carrying every class of the family on
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
# Input columns (long form): period (key, used for the file name), period_label
# (French label for the title), year, cell, series (class name), series_label
# (legend), prob (forecast probability), event (0/1 observed occurrence).
# Usage: Rscript zone_diagrams.R <pairs.csv> <out_dir> <label> [n_boot] [kind] [family]
# =============================================================================

suppressPackageStartupMessages(library(verification))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript zone_diagrams.R <pooled_pairs.csv> <out_dir> <label> [n_boot]\n")
  quit(status = 1)
}
PAIRS <- args[1]; OUT <- args[2]; LABEL <- args[3]
N_BOOT <- if (length(args) > 3) as.integer(args[4]) else 200
# "Raw" for the uncalibrated hindcast, "Calibrated" from phase P3 on: the word
# belongs in the title, not only in the folder name.
KIND <- if (length(args) > 4) args[5] else "Raw"
# name of the family, printed in the title ("terciles", "classes SPI", ...)
FAMILY <- if (length(args) > 5) args[6] else "terciles"
dir.create(file.path(OUT, "reliability"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT, "roc"), recursive = TRUE, showWarnings = FALSE)

# up to six classes per family; the first three colours are those the bulletins
# use for Below / Near / Above Normal, so a tercile figure keeps its usual look
SERIES_COL <- c("#E07B39", "#4CAF50", "#2196F3", "#8E44AD", "#16A085", "#C0392B")
SPARSE_COL <- "grey55"
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
sharp_panel <- function(tabs, names, cols) {
  cmax <- max(unlist(lapply(tabs, function(t) max(t$n_pairs, na.rm = TRUE))), na.rm = TRUE)
  plot(NA, xlim = c(0, 1), ylim = c(0, cmax * 1.12), xaxs = "i", yaxs = "i",
       xlab = "Probabilité prévue", ylab = "Effectif", cex.lab = 1.0, cex.axis = 0.9,
       main = sprintf("Netteté — %d couples par classe (%d bins)",
                      sum(tabs[[1]]$n_pairs, na.rm = TRUE), nrow(tabs[[1]])),
       font.main = 1, cex.main = 0.85)
  grid(nx = NA, ny = NULL, col = "grey90")
  nb <- nrow(tabs[[1]]); bw <- 1 / nb; sub <- bw / (length(names) + 0.6)
  for (ib in seq_len(nb)) for (k in seq_along(names)) {
    t <- tabs[[k]]
    if (!is.finite(t$n_pairs[ib]) || t$n_pairs[ib] <= 0) next
    xa <- (ib - 1) * bw + (k - 1) * sub + sub * 0.35
    rect(xa, 0, xa + sub * 0.85, t$n_pairs[ib],
         col = if (t$sparse[ib] == 0) cols[k] else SPARSE_COL, border = NA)
  }
  abline(h = 0, col = "grey40")
  box()
}

# Every class of the family shares one set of axes (one figure per period), so
# the reader sees at a glance whether the overconfidence of "below normal" is the
# same as that of "above normal", or whether the 300 mm threshold behaves like
# the 500 mm one. The no-skill line is drawn for the **mean base rate** of the
# family, and each class carries its own climatology as a dotted line in its own
# colour: in a family of percentile thresholds these differ (0.2, 0.5, 0.8).
rel_figure <- function(tabs, main, names, cols, clims) {
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i", xlab = "Probabilité prévue",
       ylab = "Fréquence observée", main = main, font.main = 2, cex.main = 1.0,
       cex.lab = 1.0, cex.axis = 0.9)
  # skill region of the attributes diagram: a bin helps the Brier skill score
  # when it lies beyond the no-skill line (x + clim)/2, away from climatology.
  clim_ref <- mean(clims, na.rm = TRUE)
  xs <- seq(0, 1, length.out = 200); ns <- (xs + clim_ref) / 2
  rgt <- xs >= clim_ref; lft <- xs <= clim_ref
  polygon(c(xs[rgt], rev(xs[rgt])), c(ns[rgt], rep(1, sum(rgt))),
          col = adjustcolor("green", alpha.f = 0.08), border = NA)
  polygon(c(xs[lft], rev(xs[lft])), c(ns[lft], rep(0, sum(lft))),
          col = adjustcolor("green", alpha.f = 0.08), border = NA)
  lines(xs, ns, col = "grey55", lwd = 1)
  for (k in seq_along(clims))
    if (is.finite(clims[k])) abline(h = clims[k], col = adjustcolor(cols[k], alpha.f = 0.5), lty = 3)
  abline(v = clim_ref, col = "grey65", lty = 3)
  abline(0, 1, lty = 2, lwd = 1.3); grid(col = "grey90")

  cmax <- max(unlist(lapply(tabs, function(t) max(t$n_pairs, na.rm = TRUE))), na.rm = TRUE)
  for (k in seq_along(names)) {
    col <- cols[k]
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
         col = cols, pt.bg = cols, legend = names)
  legend("left", bty = "n", cex = 0.66, inset = c(0, 0.16),
         legend = c("bin sous-peuplé", "IC 95 % (bloc-année)",
                    "zone de BSS positif", "climatologie de la classe"),
         pch = c(4, NA, 22, NA), lty = c(NA, 1, NA, 3), lwd = c(2, 1.4, NA, 1.4),
         col = c(SPARSE_COL, SPARSE_COL, "grey60", "grey50"),
         pt.bg = c(NA, NA, adjustcolor("green", alpha.f = 0.18), NA))

}

roc_figure <- function(rocs, envs, areas, main, names, cols) {
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), asp = 1, xlab = "Taux de fausses alertes",
       ylab = "Taux de détection", main = main, font.main = 2, cex.main = 1.0,
       cex.lab = 1.0, cex.axis = 0.9)
  abline(0, 1, lty = 2); grid(col = "grey90")
  for (k in seq_along(names)) {
    col <- cols[k]
    env <- envs[[k]]
    if (!is.null(env)) {
      ok <- is.finite(env$lo) & is.finite(env$hi)
      polygon(c(env$grid[ok], rev(env$grid[ok])), c(env$lo[ok], rev(env$hi[ok])),
              col = adjustcolor(col, alpha.f = 0.14), border = NA)
    }
    r <- rocs[[k]]; o <- order(r$FAR, r$HR)
    lines(r$FAR[o], r$HR[o], col = col, lwd = 2.4)
  }
  leg <- sapply(seq_along(names), function(k) {
    a <- areas[[k]]
    if (is.finite(a$lo)) sprintf("%s : AUC %.3f [%.3f, %.3f]", names[k], a$auc, a$lo, a$hi)
    else sprintf("%s : AUC %.3f", names[k], a$auc)
  })
  legend("bottomright", bty = "n", cex = 0.78, lwd = 2.4, col = cols, legend = leg)
  legend("topleft", bty = "n", cex = 0.7,
         legend = sprintf("N = %d (%d ans)  •  enveloppe IC 95 %% bloc-année",
                          areas[[1]]$n, areas[[1]]$n_years))
}

# --------------------------------------------------------------------- main
pairs <- read.csv(PAIRS, stringsAsFactors = FALSE)
needed <- c("period", "year", "series", "prob", "event")
if (!all(needed %in% names(pairs))) {
  cat(sprintf("zone_diagrams.R: %s — colonnes %s manquantes : aucun diagramme\n",
              LABEL, paste(setdiff(needed, names(pairs)), collapse = ", ")))
  quit(status = 0)
}
pairs <- pairs[is.finite(pairs$prob) & is.finite(pairs$event), ]
if (!nrow(pairs)) {
  cat(sprintf("zone_diagrams.R: %s — aucune paire exploitable\n", LABEL))
  quit(status = 0)
}
# the order of the classes is the order they first appear in, which is the order
# the Python side writes them: BN, NN, AN — or 300, 500, 600 mm
SERIES <- unique(pairs$series)
LABELS <- sapply(SERIES, function(sv) {
  lb <- pairs$series_label[pairs$series == sv]
  if (length(lb) && !is.na(lb[1]) && nzchar(lb[1])) as.character(lb[1]) else sv
})
COLS <- SERIES_COL[((seq_along(SERIES) - 1) %% length(SERIES_COL)) + 1]

summary_rows <- list(); bin_rows <- list()
for (per in unique(pairs$period)) {
  dper <- pairs[pairs$period == per, ]
  if (nrow(dper) < 50) next

  tabs <- list(); rocs <- list(); areas <- list(); envs <- list(); clims <- numeric(0)
  for (k in seq_along(SERIES)) {
    d <- dper[dper$series == SERIES[k], ]
    if (!nrow(d)) { tabs[[k]] <- bin_table(numeric(0), numeric(0), numeric(0)); next }
    ev <- as.integer(d$event); pr <- d$prob
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
    clims[k] <- mean(ev)
    bin_rows[[length(bin_rows) + 1]] <- data.frame(label = LABEL, family = FAMILY,
                                                   period = per, series = SERIES[k], tab)
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      label = LABEL, family = FAMILY, period = per, series = SERIES[k], n_pairs = area$n,
      n_years = area$n_years, base_rate = round(mean(ev), 6),
      roc_area = round(area$auc, 6), roc_pvalue = round(area$p, 6),
      roc_lo = round(area$lo, 6), roc_hi = round(area$hi, 6),
      resolution = round(sum(tab$n_pairs * (tab$obs_freq - mean(ev))^2, na.rm = TRUE) / sum(tab$n_pairs), 6),
      reliability = round(sum(tab$n_pairs * (tab$mean_frcst - tab$obs_freq)^2, na.rm = TRUE) / sum(tab$n_pairs), 6),
      stringsAsFactors = FALSE)
  }
  if (!length(areas)) next
  per_label <- if ("period_label" %in% names(dper)) as.character(dper$period_label[1]) else per
  sub <- sprintf("%s — %s", LABEL, per_label)

  png_open(file.path(OUT, "reliability", sprintf("%s.png", per)), w = 1150, h = 1400)
  op <- par(oma = c(0, 0, 3.2, 0))
  layout(matrix(c(1, 2), nrow = 2), heights = c(3.1, 1))
  par(mar = c(4.2, 4.4, 1.2, 1.4))
  rel_figure(tabs, "", LABELS, COLS, clims)
  par(mar = c(4.2, 4.4, 2.2, 1.4))
  sharp_panel(tabs, LABELS, COLS)
  mtext(sprintf("%s — Diagramme de fiabilité (attributs), %s", KIND, FAMILY), outer = TRUE,
        font = 2, cex = 1.0, line = 1.2)
  mtext(sub, outer = TRUE, cex = 0.85, line = 0.0)
  layout(1); par(op); dev.off()

  png_open(file.path(OUT, "roc", sprintf("%s.png", per)), w = 1100, h = 1100)
  op <- par(mar = c(4.2, 4.2, 4.0, 1.0))
  roc_figure(rocs, envs, areas, sprintf("%s — Diagramme ROC, %s", KIND, FAMILY), LABELS, COLS)
  mtext(sub, side = 3, line = 0.4, cex = 0.85)
  par(op); dev.off()
}

if (length(summary_rows))
  write.csv(do.call(rbind, summary_rows), file.path(OUT, "diagram_scores.csv"), row.names = FALSE)
if (length(bin_rows))
  write.csv(do.call(rbind, bin_rows), file.path(OUT, "diagram_bins.csv"), row.names = FALSE)
cat(sprintf("zone_diagrams.R: %s — %d figure(s)\n", LABEL,
            length(list.files(OUT, pattern = "[.]png$", recursive = TRUE))))
