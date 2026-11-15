# Regresión lineal simple y múltiple con lm(), y logística con glm().
# Réplica de notebooks/03_regresion.ipynb sobre el mismo conjunto de entrenamiento.

source(file.path("r", "00_comun.R"))
suppressPackageStartupMessages({ library(broom); library(car); library(patchwork) })

cats <- c("estu_genero", "cole_naturaleza", "cole_area_ubicacion", "cole_jornada")
tr <- read_parquet(file.path("data", "processed", "train.parquet")) |>
  filter(if_all(all_of(cats), ~ .x != "Sin dato")) |>
  select(punt_global, alto_desempeno, estu_inse_individual, educ_max_padres, indice_tic, extraedad, all_of(cats)) |>
  na.omit() |>
  mutate(jornada = relevel(factor(ifelse(cole_jornada %in% c("MAÑANA", "TARDE", "UNICA", "COMPLETA"), cole_jornada, "NOCHE O SABATINA")), ref = "MAÑANA"),
         estu_genero = relevel(factor(estu_genero), ref = "F"),
         cole_naturaleza = relevel(factor(cole_naturaleza), ref = "OFICIAL"),
         cole_area_ubicacion = relevel(factor(cole_area_ubicacion), ref = "RURAL"))
cat("n de entrenamiento con datos completos:", nrow(tr), "\n")

# --- Regresión lineal simple -----------------------------------------------------------------
m1 <- lm(punt_global ~ estu_inse_individual, data = tr)
print(summary(m1))

# --- Regresión lineal múltiple ---------------------------------------------------------------
m2 <- lm(punt_global ~ estu_inse_individual + educ_max_padres + indice_tic + extraedad +
           estu_genero + cole_naturaleza + cole_area_ubicacion + jornada, data = tr)
print(summary(m2))
coefs <- tidy(m2, conf.int = TRUE)
write.csv(coefs, file.path(TAB_DIR, "R_reg_multiple_coef.csv"), row.names = FALSE)
ajuste <- bind_rows(glance(m1) |> mutate(modelo = "simple"), glance(m2) |> mutate(modelo = "multiple")) |>
  select(modelo, r.squared, adj.r.squared, sigma, statistic, df, df.residual, nobs)
write.csv(ajuste, file.path(TAB_DIR, "R_reg_ajuste.csv"), row.names = FALSE)
print(as.data.frame(ajuste), digits = 6)
cat("VIF generalizado:\n"); print(round(vif(m2), 3))
cat("Prueba de varianza no constante (Breusch-Pagan, car::ncvTest):\n"); print(ncvTest(m2))

# --- Regresión logística ---------------------------------------------------------------------
m3 <- glm(alto_desempeno ~ estu_inse_individual + educ_max_padres + indice_tic + extraedad +
            estu_genero + cole_naturaleza + cole_area_ubicacion + jornada, data = tr, family = binomial)
orr <- tidy(m3, exponentiate = TRUE, conf.int = FALSE)
write.csv(orr, file.path(TAB_DIR, "R_reg_logistica_or.csv"), row.names = FALSE)
print(as.data.frame(orr), digits = 4)

# --- Figura R_13: recta de regresión simple sobre densidad ------------------------------------
p13 <- ggplot(tr, aes(estu_inse_individual, punt_global)) +
  geom_hex(bins = 45) +
  scale_fill_gradient(low = "#C6DBEF", high = "#08306B", trans = "log10", name = "Estudiantes") +
  geom_smooth(method = "lm", formula = y ~ x, colour = PALETA[2], linewidth = 0.9, se = FALSE) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 3), colour = PALETA[4], linewidth = 0.9, se = FALSE, linetype = "dashed") +
  coord_cartesian(ylim = c(100, 420)) +
  labs(title = sprintf("lm(): puntaje global ~ INSE (R² = %.3f)", summary(m1)$r.squared),
       subtitle = "Recta (naranja) y polinomio de grado 3 (rojo, discontinuo)",
       x = "Índice socioeconómico INSE (0–100)", y = "Puntaje global (0–500)", caption = FUENTE)
guardar(p13, "R_fig_13_lm_simple", alto = 4)

# --- Figura R_14: diagnóstico del modelo múltiple con ggplot2 ---------------------------------
set.seed(SEED)
d <- augment(m2) |> slice_sample(n = 20000)
g1 <- ggplot(d, aes(.fitted, .resid)) + geom_point(alpha = 0.08, size = 0.3, colour = PALETA[1]) +
  geom_hline(yintercept = 0) + geom_smooth(method = "loess", formula = y ~ x, colour = PALETA[4], se = FALSE, linewidth = 0.7) +
  labs(title = "Residuos vs. ajustados", x = "Valor ajustado", y = "Residuo")
g2 <- ggplot(d, aes(sample = .std.resid)) + stat_qq(size = 0.3, alpha = 0.3, colour = PALETA[1]) + stat_qq_line(colour = PALETA[4]) +
  labs(title = "Q-Q de residuos", x = "Cuantiles teóricos", y = "Residuo estandarizado")
g3 <- ggplot(d, aes(.fitted, sqrt(abs(.std.resid)))) + geom_point(alpha = 0.08, size = 0.3, colour = PALETA[1]) +
  geom_smooth(method = "loess", formula = y ~ x, colour = PALETA[4], se = FALSE, linewidth = 0.7) +
  labs(title = "Escala-localización", x = "Valor ajustado", y = expression(sqrt("|residuo estandarizado|")))
p14 <- (g1 | g2 | g3) + plot_annotation(title = "Diagnóstico de lm() múltiple (muestra de 20 000)", caption = FUENTE)
guardar(p14, "R_fig_14_diagnostico_lm", ancho = 10, alto = 3.4)

# --- Figura R_15: coeficientes con intervalo de confianza -------------------------------------
etq <- c(estu_inse_individual = "INSE (por punto)", educ_max_padres = "Educación de los padres (por nivel)", indice_tic = "Índice TIC (por recurso)",
         extraedad = "Extraedad", estu_generoM = "Hombre", `cole_naturalezaNO OFICIAL` = "Colegio no oficial", cole_area_ubicacionURBANO = "Zona urbana",
         jornadaCOMPLETA = "Jornada completa", `jornadaNOCHE O SABATINA` = "Jornada nocturna o sabatina", jornadaTARDE = "Jornada de la tarde", jornadaUNICA = "Jornada única")
p15 <- coefs |> filter(term != "(Intercept)") |> mutate(termino = etq[term]) |>
  ggplot(aes(estimate, reorder(termino, estimate))) + geom_vline(xintercept = 0) +
  geom_pointrange(aes(xmin = conf.low, xmax = conf.high), colour = PALETA[1], size = 0.35) +
  labs(title = "Coeficientes de lm() con IC 95 % (broom + ggplot2)", x = "Puntos del puntaje global", y = NULL, caption = FUENTE)
guardar(p15, "R_fig_15_coeficientes_lm", alto = 3.6)
