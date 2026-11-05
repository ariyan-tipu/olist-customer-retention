#!/usr/bin/env Rscript
# Survival analysis of customer repeat purchase on the Olist dataset.
#
# The question: after a customer's first order arrives, how long until they
# buy again, and how does that first experience (late delivery, review
# score) change the hazard of returning?
#
# Censoring is the whole point. Most customers have not purchased again by
# the end of the observation window. They are not churned, they are
# right-censored: we know only that their time to next purchase exceeds
# what we observed. A classifier has to discard or mislabel them; the
# survival machinery uses exactly the information they do carry.
#
# Inputs:  the survival dataset exported by the Spark pipeline.
# Outputs: Kaplan-Meier plots; Cox hazard ratios with diagnostics; a
#          Weibull AFT fit whose parameters are exported so the MATLAB
#          script can re-derive them by direct likelihood maximisation.
#
# Only base R and the survival package are used (survival ships with R).

suppressPackageStartupMessages(library(survival))

DAYS_PLOT_LIMIT <- 540

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  input <- if (length(args) >= 1) args[1] else file.path("outputs", "survival", "survival_dataset.csv")
  out_dir <- if (length(args) >= 2) args[2] else file.path("outputs", "survival")
  list(input = input, out_dir = out_dir)
}

read_spells <- function(path) {
  spells <- read.csv(path, stringsAsFactors = FALSE)
  required <- c(
    "time_days", "event", "delivery_delay_days", "late_delivery",
    "review_score", "order_value", "n_items", "installments", "customer_region"
  )
  missing <- setdiff(required, names(spells))
  if (length(missing) > 0) {
    stop("survival dataset is missing columns: ", paste(missing, collapse = ", "))
  }

  # Review score grouped with an explicit level for missing reviews, so
  # customers who never reviewed stay in the model instead of being
  # dropped by complete-case handling.
  spells$review_group <- with(spells, ifelse(
    is.na(review_score), "none",
    ifelse(review_score <= 2, "low", ifelse(review_score == 3, "mid", "high"))
  ))
  spells$review_group <- factor(spells$review_group, levels = c("high", "mid", "low", "none"))
  spells$late_delivery <- as.integer(spells$late_delivery)
  spells$customer_region <- stats::relevel(factor(spells$customer_region), ref = "sudeste")
  spells$log_order_value <- log1p(spells$order_value)
  spells
}

km_plot <- function(spells, group_formula, colours, labels, title, path) {
  fit <- survfit(group_formula, data = spells)
  grDevices::png(path, width = 900, height = 650, res = 120)
  # fun = "event" plots cumulative incidence of repurchase, which reads
  # more naturally than survival when the event is a good outcome.
  plot(
    fit, fun = "event", col = colours, lwd = 2, conf.int = TRUE,
    xlim = c(0, DAYS_PLOT_LIMIT),
    xlab = "Days since first delivery",
    ylab = "Cumulative share who purchased again",
    main = title
  )
  legend("topleft", legend = labels, col = colours, lwd = 2, bty = "n")
  grDevices::dev.off()
  invisible(fit)
}

fit_cox <- function(spells) {
  coxph(
    Surv(time_days, event) ~ late_delivery + review_group + log_order_value +
      n_items + installments + customer_region,
    data = spells
  )
}

hazard_ratio_table <- function(cox_fit) {
  s <- summary(cox_fit)
  data.frame(
    covariate = rownames(s$conf.int),
    hazard_ratio = round(unname(s$conf.int[, "exp(coef)"]), 4),
    ci_lower = round(unname(s$conf.int[, "lower .95"]), 4),
    ci_upper = round(unname(s$conf.int[, "upper .95"]), 4),
    p_value = signif(unname(s$coefficients[, "Pr(>|z|)"]), 4),
    stringsAsFactors = FALSE
  )
}

fit_weibull_aft <- function(spells) {
  # Deliberately compact model: the two covariates with the clearest
  # story, on complete cases, so the MATLAB script can re-derive the
  # same fit from the raw likelihood without a design-matrix library.
  aft_data <- spells[!is.na(spells$review_score), ]
  aft_data$low_review <- as.integer(aft_data$review_score <= 2)
  fit <- survreg(
    Surv(time_days, event) ~ late_delivery + low_review,
    data = aft_data, dist = "weibull"
  )
  list(fit = fit, data = aft_data)
}

export_aft_for_matlab <- function(aft, out_dir) {
  inputs <- aft$data[, c("time_days", "event", "late_delivery", "low_review")]
  write.csv(inputs, file.path(out_dir, "aft_inputs.csv"), row.names = FALSE)

  fit <- aft$fit
  params <- data.frame(
    intercept = unname(coef(fit)["(Intercept)"]),
    coef_late_delivery = unname(coef(fit)["late_delivery"]),
    coef_low_review = unname(coef(fit)["low_review"]),
    log_scale = log(fit$scale),
    loglik = fit$loglik[2],
    n = nrow(aft$data),
    events = sum(aft$data$event)
  )
  write.csv(params, file.path(out_dir, "aft_params_r.csv"), row.names = FALSE)
  params
}

survival_summaries <- function(spells, out_dir) {
  fit <- survfit(Surv(time_days, event) ~ late_delivery, data = spells)
  horizons <- c(90, 180, 365)
  s <- summary(fit, times = horizons, extend = TRUE)
  data.frame(
    group = s$strata,
    days = s$time,
    repurchase_share = round(1 - s$surv, 5),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  args <- parse_args()
  dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)
  spells <- read_spells(args$input)

  cat(sprintf(
    "Spells: %d, events: %d, censoring rate: %.1f%%\n",
    nrow(spells), sum(spells$event), 100 * (1 - mean(spells$event))
  ))

  km_plot(
    spells, Surv(time_days, event) ~ late_delivery,
    colours = c("#2c7fb8", "#d95f0e"),
    labels = c("Delivered on time", "Delivered late"),
    title = "Repeat purchase after first delivery",
    path = file.path(args$out_dir, "km_late_delivery.png")
  )
  km_plot(
    spells, Surv(time_days, event) ~ review_group,
    colours = c("#1b9e77", "#7570b3", "#d95f02", "#999999"),
    labels = c("Review 4-5", "Review 3", "Review 1-2", "No review"),
    title = "Repeat purchase by first review score",
    path = file.path(args$out_dir, "km_review_group.png")
  )

  cox_fit <- fit_cox(spells)
  hr <- hazard_ratio_table(cox_fit)
  write.csv(hr, file.path(args$out_dir, "cox_hazard_ratios.csv"), row.names = FALSE)
  cat("\nCox proportional hazards (hazard of repurchasing):\n")
  print(hr, row.names = FALSE)

  zph <- cox.zph(cox_fit)
  zph_table <- data.frame(
    term = rownames(zph$table),
    chisq = round(zph$table[, "chisq"], 3),
    df = zph$table[, "df"],
    p_value = signif(zph$table[, "p"], 4)
  )
  write.csv(zph_table, file.path(args$out_dir, "cox_ph_diagnostics.csv"), row.names = FALSE)
  cat("\nProportional hazards diagnostics (small p suggests violation):\n")
  print(zph_table, row.names = FALSE)

  aft <- fit_weibull_aft(spells)
  params <- export_aft_for_matlab(aft, args$out_dir)
  cat("\nWeibull AFT (exported for the MATLAB cross-check):\n")
  print(params, row.names = FALSE)

  km_summary <- survival_summaries(spells, args$out_dir)
  write.csv(km_summary, file.path(args$out_dir, "km_summary.csv"), row.names = FALSE)
  cat("\nCumulative repurchase share at fixed horizons:\n")
  print(km_summary, row.names = FALSE)

  cat("\nSurvival analysis complete. Outputs in ", args$out_dir, "\n", sep = "")
}

main()
