# Configuración compartida por los scripts R del proyecto.
# Ejecutar desde la raíz: Rscript --vanilla r/01_descriptiva.R

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(ggplot2)
})

SEED <- 42
set.seed(SEED)

FIG_DIR <- file.path("reports", "figures")
TAB_DIR <- file.path("reports", "tables")
FUENTE <- "Fuente: ICFES, Saber 11° 2020-2 (datos.gov.co). Elaboración propia."
# Misma paleta Okabe-Ito que en Python.
PALETA <- c("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")

ORDEN_ESTRATO <- c("Sin Estrato", paste("Estrato", 1:6))
PUNTAJES_AREA <- c("punt_lectura_critica", "punt_matematicas", "punt_c_naturales",
                   "punt_sociales_ciudadanas", "punt_ingles")

leer_limpio <- function() {
  read_parquet(file.path("data", "processed", "saber11_2020-2_clean.parquet")) |>
    mutate(fami_estratovivienda = factor(as.character(fami_estratovivienda), levels = ORDEN_ESTRATO))
}

leer_muestra <- function() {
  read_parquet(file.path("data", "processed", "saber11_2020-2_muestra.parquet")) |>
    mutate(fami_estratovivienda = factor(as.character(fami_estratovivienda), levels = ORDEN_ESTRATO))
}

tema <- theme_minimal(base_size = 10) +
  theme(plot.title = element_text(face = "bold", size = 11),
        plot.caption = element_text(size = 7, colour = "#555555", hjust = 0),
        panel.grid.minor = element_blank())
theme_set(tema)

guardar <- function(p, nombre, ancho = 6, alto = 3.8) {
  ggsave(file.path(FIG_DIR, paste0(nombre, ".png")), p, width = ancho, height = alto, dpi = 300, bg = "white")
}

moda <- function(x) {
  x <- x[!is.na(x)]
  as.numeric(names(which.max(table(x))))
}
