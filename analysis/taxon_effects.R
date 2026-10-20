# Do some taxa get misclassified more than others, and is that phylogenetic?
#
#   Rscript analysis/taxon_effects.R results/taxa/<run>_<splits>_sheets.csv [tree.tre]
#
# Input is the per-sheet, per-organ table from tools/by_taxon.py. Run from the
# repo root: training sheets per taxon come from splits/taxonomy.csv.
#
# Rare genera are rare in training too, so "hard genus" and "little training
# data" are confounded. Both parts therefore adjust for log(1 + training
# sheets) of the taxon, and report what is left over.
#
# 1. Mixed model of classification error:
#      error ~ organ * truth + log_train + (1 | genus) + (1 | genus:organ)
#              + (1 | genus:species) + (1 | image_id)
#    organ * truth separates misses from false alarms, so a genus with more
#    (or fewer) positives is not called hard just for its prevalence; this
#    is why F1, which moves with prevalence, can differ between genera when
#    error odds do not. image_id absorbs sheet-level difficulty shared by the three organs;
#    genus:organ lets a genus be hard for one organ only (e.g. buds). A
#    likelihood-ratio test against the model without both genus terms asks
#    whether genera differ beyond species and sheet noise. Genus-by-organ
#    effects are printed as odds ratios with sheet counts (rare genera are
#    shrunk toward the mean).
# 2. With a tree (Newick; tips Genus_species, or Genus for a genus-level
#    tree): phylogenetic signal in error rates, as Pagel's lambda and
#    Blomberg's K on the residuals of empirical-logit error rates regressed
#    (inverse-variance weighted) on log training sheets, with their sampling
#    variance passed as measurement error (phytools::phylosig se=). Signal in
#    the raw rates is printed alongside for comparison.
suppressPackageStartupMessages({
  library(lme4)
  library(ape)
  library(phytools)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript analysis/taxon_effects.R <sheets.csv> [tree.tre]")
sheets <- read.csv(args[1], stringsAsFactors = FALSE)
sheets$organ <- factor(sheets$organ, levels = c("bud", "flower", "fruit"))
taxonomy <- read.csv("splits/taxonomy.csv", stringsAsFactors = FALSE)
train <- taxonomy[taxonomy$split == "train", ]
train$species[train$species == ""] <- paste(train$genus[train$species == ""], "sp.")
log_train <- function(units, train_units) log1p(as.numeric(table(factor(train_units, levels = unique(units)))[units]))
# Centred, so the intercept stays at a genus of typical training size.
sheets$log_train <- log_train(sheets$genus, train$genus)
sheets$log_train <- sheets$log_train - mean(sheets$log_train)
cat(sprintf("%d sheets, %d genera, %d species, %d sheet-organ calls, %d errors\n\n",
            length(unique(sheets$image_id)), length(unique(sheets$genus)),
            length(unique(sheets$species)), nrow(sheets), sum(sheets$error)))

# 1. Mixed model --------------------------------------------------------------
control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
full <- glmer(error ~ organ * truth + log_train + (1 | genus) + (1 | genus:organ) + (1 | genus:species) + (1 | image_id),
              data = sheets, family = binomial, control = control)
no_genus <- glmer(error ~ organ * truth + log_train + (1 | genus:species) + (1 | image_id),
                  data = sheets, family = binomial, control = control)
cat("Random-effect SDs (logit scale):\n")
print(as.data.frame(VarCorr(full))[, c("grp", "sdcor")], row.names = FALSE)
size <- summary(full)$coefficients["log_train", ]
cat(sprintf("\nlog(1 + training sheets of the genus): %.3f per unit (error odds x%.2f per e-fold more data), z = %.2f, p = %.4f\n",
            size["Estimate"], exp(size["Estimate"]), size["z value"], size["Pr(>|z|)"]))
lrt <- anova(no_genus, full)
# Two variances tested on the boundary: the chi-square(2) p-value is conservative.
cat(sprintf("Genus effects beyond training size (genus + genus:organ): LRT chi2 = %.2f, df = %d, p = %.4f\n\n",
            lrt$Chisq[2], lrt$Df[2], lrt$`Pr(>Chisq)`[2]))

# Genus-by-organ odds ratio = exp(genus mode + genus:organ mode).
re <- ranef(full)
genus_mode <- setNames(re$genus[, 1], rownames(re$genus))
cell_mode <- setNames(re$`genus:organ`[, 1], rownames(re$`genus:organ`))
cells <- unique(sheets[, c("genus", "organ")])
cells$sheets <- mapply(function(g, o) sum(sheets$genus == g & sheets$organ == o), cells$genus, cells$organ)
cells$errors <- mapply(function(g, o) sum(sheets$error[sheets$genus == g & sheets$organ == o]), cells$genus, cells$organ)
cells$odds_ratio <- exp(genus_mode[cells$genus] + cell_mode[paste(cells$genus, cells$organ, sep = ":")])
cells <- cells[order(cells$organ, -cells$odds_ratio), ]
cat("Genus-by-organ effects on error odds beyond training size (conditional modes; 1 = average genus):\n")
print(format(cells, digits = 3), row.names = FALSE)

# 2. Phylogenetic signal ------------------------------------------------------
if (length(args) < 2) {
  cat("\nNo tree given; skipping phylogenetic signal.\n")
  quit(save = "no")
}
tree <- read.tree(args[2])
genus_level <- !any(grepl("_", tree$tip.label))
unit <- if (genus_level) sheets$genus else gsub(" ", "_", sheets$species)
unit[grepl("_sp\\.$", unit)] <- NA  # genus-only identifications have no species tip
agg <- aggregate(error ~ unit, data = data.frame(unit = unit, error = sheets$error),
                 FUN = function(e) c(errors = sum(e), calls = length(e)))
agg <- data.frame(unit = agg$unit, errors = agg$error[, "errors"], calls = agg$error[, "calls"])
agg <- agg[agg$unit %in% tree$tip.label, ]
cat(sprintf("\n%s-level tree: %d of %d units with data are tips\n",
            if (genus_level) "Genus" else "Species", nrow(agg), length(unique(na.omit(unit)))))
if (nrow(agg) < 10) {
  cat("Fewer than 10 matched tips; too few for a signal test.\n")
  quit(save = "no")
}
pruned <- keep.tip(tree, agg$unit)
logit <- setNames(log((agg$errors + 0.5) / (agg$calls - agg$errors + 0.5)), agg$unit)
se <- setNames(sqrt(1 / (agg$errors + 0.5) + 1 / (agg$calls - agg$errors + 0.5)), agg$unit)
train_units <- if (genus_level) train$genus else gsub(" ", "_", train$species)
agg$log_train <- log_train(agg$unit, train_units)
size_fit <- lm(logit ~ log_train, data = cbind(agg, logit = logit), weights = 1 / se^2)
cat(sprintf("Weighted regression on log training sheets: slope %.3f (p = %.4f)\n",
            coef(size_fit)[2], summary(size_fit)$coefficients[2, 4]))
residual <- setNames(residuals(size_fit), agg$unit)
for (target in list(list("raw error rates", logit), list("residuals after training size", residual))) {
  lambda <- phylosig(pruned, target[[2]], method = "lambda", se = se, test = TRUE)
  K <- phylosig(pruned, target[[2]], method = "K", se = se, test = TRUE, nsim = 999)
  cat(sprintf("%-30s Pagel's lambda = %.3f (p = %.4f)   Blomberg's K = %.3f (p = %.4f)\n",
              target[[1]], lambda$lambda, lambda$P, K$K, K$P))
}
