# PCA y clustering en R: prcomp(), kmeans(), hclust().
# Réplica de notebooks/05_multivariante.ipynb.

source(file.path("r", "00_comun.R"))
suppressPackageStartupMessages(library(factoextra))

df <- leer_limpio() |> filter(if_all(all_of(PUNTAJES_AREA), ~ !is.na(.x)))
X <- scale(as.matrix(df[, PUNTAJES_AREA]))
colnames(X) <- c("Lectura", "Matemáticas", "Naturales", "Sociales", "Inglés")

# --- PCA ----------------------------------------------------------------------------------
pca <- prcomp(X, center = FALSE, scale. = FALSE)
varianza <- data.frame(componente = paste0("CP", 1:5), autovalor = pca$sdev^2,
                       varianza_pct = 100 * pca$sdev^2 / sum(pca$sdev^2))
varianza$acumulada_pct <- cumsum(varianza$varianza_pct)
write.csv(varianza, file.path(TAB_DIR, "R_mv_pca_varianza.csv"), row.names = FALSE)
print(varianza, digits = 4)
print(round(pca$rotation, 3))

set.seed(SEED)
sub <- sample(nrow(X), 5000)
pca_sub <- prcomp(X[sub, ], center = FALSE, scale. = FALSE)
p9 <- fviz_pca_biplot(pca_sub, geom.ind = "point", pointsize = 0.4, alpha.ind = 0.25, col.ind = "#888888",
                      col.var = PALETA[4], repel = TRUE, labelsize = 3) +
  labs(title = "Biplot de componentes principales (factoextra, submuestra de 5 000)", caption = FUENTE) + tema
guardar(p9, "R_fig_09_biplot", alto = 4.2)

p10 <- fviz_eig(pca, addlabels = TRUE, barfill = PALETA[1], barcolor = PALETA[1], linecolor = PALETA[4]) +
  labs(title = "Gráfico de sedimentación (prcomp)", x = "Componente", y = "Varianza explicada (%)", caption = FUENTE) + tema
guardar(p10, "R_fig_10_sedimentacion")

# --- K-Means --------------------------------------------------------------------------------
set.seed(SEED)
# Lloyd en lugar de Hartigan-Wong (el valor por defecto): con medio millón de filas este último no termina en un tiempo razonable.
km <- kmeans(X, centers = 3, nstart = 5, iter.max = 100, algorithm = "Lloyd")
orden <- order(rowMeans(km$centers))
centros <- as.data.frame(km$centers[orden, ])
centros$n <- km$size[orden]
centros$cluster <- paste0("C", 1:3)
write.csv(centros, file.path(TAB_DIR, "R_mv_kmeans_centros.csv"), row.names = FALSE)
print(centros, digits = 3)
cat(sprintf("Varianza entre clústeres / total: %.1f %%\n", 100 * km$betweenss / km$totss))

# --- Jerárquico (Ward) ----------------------------------------------------------------------
set.seed(SEED)
sub3 <- sample(nrow(X), 3000)
hc <- hclust(dist(X[sub3, ]), method = "ward.D2")
grupos <- cutree(hc, k = 3)
print(table(jerarquico = grupos, kmeans = match(km$cluster[sub3], orden)))

# Dendrograma con gráficos base: fviz_dend() es muy lento con 3 000 hojas.
png(file.path(FIG_DIR, "R_fig_11_dendrograma.png"), width = 7, height = 3.8, units = "in", res = 300)
par(mar = c(2.5, 4, 2.5, 1), cex = 0.8)
plot(hc, labels = FALSE, hang = -1, main = "Dendrograma de Ward (hclust, submuestra de 3 000)", xlab = "", sub = "", ylab = "Distancia de Ward")
rect.hclust(hc, k = 3, border = PALETA[1:3])
mtext(FUENTE, side = 1, line = 1, adj = 0, cex = 0.6, col = "#555555")
dev.off()

# --- Clústeres en el plano CP1–CP2 -----------------------------------------------------------
vis <- data.frame(CP1 = pca$x[sub, 1], CP2 = pca$x[sub, 2], cluster = factor(match(km$cluster[sub], orden), labels = c("C1 · Bajo", "C2 · Medio", "C3 · Alto")))
p12 <- ggplot(vis, aes(CP1, CP2, colour = cluster)) +
  geom_point(alpha = 0.4, size = 0.6) +
  stat_ellipse(linewidth = 0.6) +
  scale_colour_manual(values = PALETA[1:3], name = "Clúster") +
  labs(title = "Clústeres K-Means en el plano de componentes principales (ggplot2)",
       x = sprintf("CP1 (%.1f %%)", varianza$varianza_pct[1]), y = sprintf("CP2 (%.1f %%)", varianza$varianza_pct[2]), caption = FUENTE) +
  theme(legend.position = "bottom")
guardar(p12, "R_fig_12_clusters_pca", alto = 4.2)
