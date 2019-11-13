#INSTALL AND LOAD PACKAGES-----

#If pacman is missing we install it, then we load libraries
if (!require("pacman")) {
  install.packages("pacman")
} else{
  library(pacman)
  pacman::p_load(e1071, plotly, purrr, Metrics, randomForestSRC, caTools, Rfast, DMwR, ranger, h2o, lubridate, ggplot2, RMySQL, caret, readr, dplyr, tidyr, rstudioapi)
}

#DIRECTORY -----

current_path = getActiveDocumentContext()$path
setwd(dirname(current_path))
setwd("..")
getwd()