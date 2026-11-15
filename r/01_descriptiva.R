# Estadística descriptiva y gráficos básicos en R (ggplot2).
# Replica en R los cálculos de notebooks/01_eda.ipynb para contrastar resultados y herramientas.

source(file.path("r", "00_comun.R"))
suppressPackageStartupMessages(library(patchwork))

df <- leer_limpio()
muestra <- leer_muestra()

# --- Medidas de tendencia central, dispersión y forma -------------------------------------
asimetria <- function(x) { x <- x[!is.na(x)]; n <- length(x); m <- mean(x); s <- sd(x)
  n / ((n - 1) * (n - 2)) * sum(((x - m) / s)^3) }
curtosis <- function(x) { x <- x[!is.na(x)]; n <- length(x); m <- mean(x); s <- sd(x)
  n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * sum(((x - m) / s)^4) - 3 * (n - 1)^2 / ((n - 2) * (n - 3)) }

vars <- c("punt_global", PUNTAJES_AREA, "estu_inse_individual", "edad")
desc <- do.call(rbind, lapply(vars, function(v) {
  x <- df[[v]]
  data.frame(variable = v, n = sum(!is.na(x)), media = mean(x, na.rm = TRUE), mediana = median(x, na.rm = TRUE),
             moda = moda(x), varianza = var(x, na.rm = TRUE), desv_est = sd(x, na.rm = TRUE),
             rango = diff(range(x, na.rm = TRUE)), IQR = IQR(x, na.rm = TRUE),
             asimetria = asimetria(x), curtosis = curtosis(x))
}))
write.csv(desc, file.path(TAB_DIR, "R_descriptivos.csv"), row.names = FALSE, fileEncoding = "UTF-8")
print(desc, digits = 4)

# --- Distribución de frecuencias ------------------------------------------------------------
freq <- df |> count(fami_estratovivienda) |> mutate(pct = 100 * n / sum(n), pct_acum = cumsum(pct))
write.csv(freq, file.path(TAB_DIR, "R_freq_estrato.csv"), row.names = FALSE, fileEncoding = "UTF-8")
print(freq)

# --- Figura R_01: histograma con densidad ---------------------------------------------------
p1 <- ggplot(df, aes(punt_global)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60, fill = PALETA[1], alpha = 0.85) +
  geom_density(colour = PALETA[4], linewidth = 0.7) +
  geom_vline(xintercept = mean(df$punt_global), linetype = "dashed") +
  labs(title = "Distribución del puntaje global (ggplot2)", x = "Puntaje global (0–500)", y = "Densidad", caption = FUENTE)
guardar(p1, "R_fig_01_histograma_global")

# --- Figura R_02: caja y violín por naturaleza del colegio ----------------------------------
p2 <- ggplot(muestra, aes(cole_naturaleza, punt_global, fill = cole_naturaleza)) +
  geom_violin(alpha = 0.5, colour = NA) +
  geom_boxplot(width = 0.18, outlier.size = 0.3) +
  scale_fill_manual(values = PALETA[c(1, 2)], guide = "none") +
  labs(title = "Puntaje global por naturaleza del colegio (muestra de 60 000)",
       x = "Naturaleza del colegio", y = "Puntaje global (0–500)", caption = FUENTE)
guardar(p2, "R_fig_02_violin_naturaleza")

# --- Figura R_03: barras de frecuencia por estrato ------------------------------------------
p3 <- ggplot(filter(freq, !is.na(fami_estratovivienda)), aes(fami_estratovivienda, pct)) +
  geom_col(fill = PALETA[3]) +
  geom_text(aes(label = sprintf("%.1f %%", pct)), vjust = -0.4, size = 2.8) +
  labs(title = "Distribución de estudiantes por estrato", x = "Estrato de la vivienda", y = "Estudiantes (%)", caption = FUENTE) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
guardar(p3, "R_fig_03_barras_estrato")

# --- Figura R_04: el mismo hallazgo que fig_09a/b/c de Python, ahora en ggplot2 -------------
t0 <- Sys.time()
res <- df |> filter(!is.na(fami_estratovivienda)) |> group_by(fami_estratovivienda) |>
  summarise(media = mean(punt_global), sd = sd(punt_global), n = n(), .groups = "drop") |>
  mutate(ic95 = 1.96 * sd / sqrt(n))
p4 <- ggplot(res, aes(fami_estratovivienda, media)) +
  geom_col(fill = PALETA[5]) +
  geom_errorbar(aes(ymin = media - ic95, ymax = media + ic95), width = 0.25) +
  coord_cartesian(ylim = c(0, 320)) +
  labs(title = "ggplot2 · Puntaje global medio por estrato (IC 95 %)",
       x = "Estrato de la vivienda", y = "Puntaje global (0–500)", caption = FUENTE) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
guardar(p4, "R_fig_04_estrato_ggplot2")
t_barras <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# --- Figura R_05: dispersión INSE vs. global con recta de regresión -------------------------
p5 <- ggplot(muestra, aes(estu_inse_individual, punt_global)) +
  geom_point(alpha = 0.08, size = 0.4, colour = PALETA[1], na.rm = TRUE) +
  geom_smooth(method = "lm", colour = PALETA[4], linewidth = 0.8, na.rm = TRUE) +
  labs(title = "Nivel socioeconómico y puntaje global (muestra de 60 000)",
       x = "Índice socioeconómico INSE (0–100)", y = "Puntaje global (0–500)", caption = FUENTE)
guardar(p5, "R_fig_05_dispersion_inse")

# --- Rendimiento: dispersión con todos los registros, comparable con la medición en Python --
d <- df |> filter(!is.na(estu_inse_individual), !is.na(punt_global))
t0 <- Sys.time()
tmp <- file.path(FIG_DIR, "_tmp_gg.png")
ggsave(tmp, ggplot(d, aes(estu_inse_individual, punt_global)) + geom_point(alpha = 0.05, size = 0.2),
       width = 5, height = 4, dpi = 150)
t_disp <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
unlink(tmp)
write.csv(data.frame(herramienta = "ggplot2", segundos = round(t_disp, 2), n = nrow(d), barras_seg = round(t_barras, 2)),
          file.path(TAB_DIR, "R_rendimiento.csv"), row.names = FALSE)
cat(sprintf("ggplot2: dispersión de %d puntos en %.2f s; barras en %.2f s\n", nrow(d), t_disp, t_barras))
