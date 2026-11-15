# Pruebas de hipótesis en R: t.test(), aov() + TukeyHSD(), chisq.test().
# Réplica de notebooks/04_hipotesis.ipynb para verificar que ambas herramientas coinciden.

source(file.path("r", "00_comun.R"))
suppressPackageStartupMessages({ library(broom); library(car); library(effectsize) })

df <- leer_limpio()
umbral <- quantile(df$punt_global, 0.75)
df$alto_desempeno <- as.integer(df$punt_global >= umbral)

# --- H1 y H2: t de Welch ------------------------------------------------------------------
t_nat <- t.test(punt_global ~ cole_naturaleza, data = df, var.equal = FALSE)
t_zona <- t.test(punt_global ~ cole_area_ubicacion, data = df, var.equal = FALSE)
d_nat <- cohens_d(punt_global ~ cole_naturaleza, data = df)
d_zona <- cohens_d(punt_global ~ cole_area_ubicacion, data = df)
lev_nat <- leveneTest(punt_global ~ factor(cole_naturaleza), data = df, center = median)
w_nat <- wilcox.test(punt_global ~ cole_naturaleza, data = df)

tt <- bind_rows(
  tidy(t_nat) |> mutate(variable = "cole_naturaleza", d_Cohen = d_nat$Cohens_d),
  tidy(t_zona) |> mutate(variable = "cole_area_ubicacion", d_Cohen = d_zona$Cohens_d)
) |> select(variable, estimate, estimate1, estimate2, statistic, parameter, p.value, conf.low, conf.high, d_Cohen)
write.csv(tt, file.path(TAB_DIR, "R_hip_t.csv"), row.names = FALSE)
print(as.data.frame(tt), digits = 5)
cat(sprintf("Levene (naturaleza): F = %.1f, p = %.3g | Wilcoxon: W = %.4g, p = %.3g\n",
            lev_nat$`F value`[1], lev_nat$`Pr(>F)`[1], w_nat$statistic, w_nat$p.value))

# --- H3: ANOVA de un factor + Tukey -------------------------------------------------------
d3 <- df |> filter(!is.na(fami_estratovivienda)) |> mutate(estrato = droplevels(fami_estratovivienda))
modelo <- aov(punt_global ~ estrato, data = d3)
an <- tidy(modelo)
eta <- eta_squared(modelo)
welch <- oneway.test(punt_global ~ estrato, data = d3, var.equal = FALSE)
kw <- kruskal.test(punt_global ~ estrato, data = d3)
an$eta2 <- c(eta$Eta2, NA)
write.csv(an, file.path(TAB_DIR, "R_hip_anova.csv"), row.names = FALSE)
print(as.data.frame(an), digits = 6)
cat(sprintf("Welch: F = %.1f, p = %.3g | Kruskal-Wallis: H = %.1f, p = %.3g\n",
            welch$statistic, welch$p.value, kw$statistic, kw$p.value))

tk <- tidy(TukeyHSD(modelo))
write.csv(tk, file.path(TAB_DIR, "R_hip_tukey.csv"), row.names = FALSE)

# --- H4 y H5: chi-cuadrado ----------------------------------------------------------------
chi_int <- chisq.test(table(df$fami_tieneinternet, df$cole_area_ubicacion), correct = FALSE)
chi_est <- chisq.test(table(d3$estrato, d3$alto_desempeno))
v <- function(x) sqrt(x$statistic / (sum(x$observed) * (min(dim(x$observed)) - 1)))
chi <- data.frame(contraste = c("internet x zona", "estrato x alto_desempeno"),
                  chi2 = c(chi_int$statistic, chi_est$statistic), gl = c(chi_int$parameter, chi_est$parameter),
                  p = c(chi_int$p.value, chi_est$p.value), V_Cramer = c(v(chi_int), v(chi_est)))
write.csv(chi, file.path(TAB_DIR, "R_hip_chi2.csv"), row.names = FALSE)
print(chi, digits = 5)

# --- Figura R_06: medias por estrato con IC 95 % -------------------------------------------
res <- d3 |> group_by(estrato) |> summarise(media = mean(punt_global), ee = sd(punt_global) / sqrt(n()), .groups = "drop")
p6 <- ggplot(res, aes(estrato, media, group = 1)) +
  geom_line(colour = "#999999") +
  geom_pointrange(aes(ymin = media - 1.96 * ee, ymax = media + 1.96 * ee), colour = PALETA[1], size = 0.4) +
  geom_hline(yintercept = mean(df$punt_global), linetype = "dashed") +
  labs(title = sprintf("Puntaje global medio por estrato · ANOVA F = %s, η² = %.3f",
                       format(round(an$statistic[1]), big.mark = " "), eta$Eta2),
       x = "Estrato de la vivienda", y = "Puntaje global (0–500)", caption = FUENTE) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
guardar(p6, "R_fig_06_medias_estrato")

# --- Figura R_07: intervalos de Tukey ------------------------------------------------------
p7 <- ggplot(tk, aes(estimate, reorder(contrast, estimate))) +
  geom_vline(xintercept = 0) +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high), colour = PALETA[1], size = 0.25) +
  labs(title = "Comparaciones de Tukey HSD entre estratos (ggplot2)",
       x = "Diferencia de medias del puntaje global (IC 95 % simultáneo)", y = NULL, caption = FUENTE) +
  theme(axis.text.y = element_text(size = 6))
guardar(p7, "R_fig_07_tukey", alto = 4.6)

# --- Figura R_08: densidades por naturaleza y zona -----------------------------------------
p8 <- ggplot(df, aes(punt_global, fill = cole_naturaleza)) +
  geom_density(alpha = 0.5, colour = NA) +
  facet_wrap(~cole_area_ubicacion) +
  scale_fill_manual(values = PALETA[c(2, 1)], name = "Colegio") +
  labs(title = "Distribución del puntaje global por naturaleza y zona del colegio",
       x = "Puntaje global (0–500)", y = "Densidad", caption = FUENTE) +
  theme(legend.position = "bottom")
guardar(p8, "R_fig_08_densidades_grupos")
