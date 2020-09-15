library(edgeR)
# based on edgeR user guide:
arab <- readRDS("arab.rds")
print(head(arab))

# Levels
Treat <- factor(substring(colnames(arab),1,4))
Treat <- relevel(Treat, ref="mock")
Time <- factor(substring(colnames(arab),5,5))

# DGEList
y <- DGEList(counts=arab, group=Treat)

# Calc filter
keep <- filterByExpr(y)
print(table(keep))

# Do the filtering
y <- y[keep, , keep.lib.sizes=FALSE]

# Export as csv
counts <- as.data.frame(y$counts)
write.csv(counts, file='arab.csv')

# Print answers
print(calcNormFactors(y)$samples)
print(calcNormFactors(counts))