#INSTALL AND LOAD PACKAGES-----

#If pacman is missing we install it, then we load libraries
if (!require("pacman")) {
  install.packages("pacman")
} else{
  library(pacman)
  pacman::p_load(kknn, e1071, gplot2, caret, readr, dplyr, tidyr, doParallel, plotly, corrplot)
}

#DIRECTORY -----

current_path = getActiveDocumentContext()$path
setwd(dirname(current_path))
setwd("..")
getwd()


#CLUSTERS -----
# Find how many cores are on your machine
detectCores() # 4

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(2)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)


#UPLOAD DATA -----

iPhone.df <- read_csv("poni6/Desktop/Data Analysis/Modulo 4/Data/iphone_smallmatrix_labeled_8d.csv")
View(iPhone.df)
str(iPhone.df)
summary(iPhone.df)

Galaxy.df <- read_csv("poni6/Desktop/Data Analysis/Modulo 4/Data/galaxy_smallmatrix_labeled_8d.csv")
View(Galaxy.df)
str(Galaxy.df)
summary(Galaxy.df)


#DATA EXPLORATION -----
#Ploting Sentiment distribution
plot_ly(iPhone.df, x= ~iPhone.df$iphonesentiment, type='histogram')
plot_ly(Galaxy.df, x= ~Galaxy.df$galaxysentiment, type='histogram')

#Checking for NAs
sapply(iPhone.df, function(x) sum(is.na(x))) #No NAs
sapply(Galaxy.df, function(x) sum(is.na(x))) #No NAs

#FEATURE ENGINEERING IPHONE ----

#CORELATION
options(max.print = 1000000)
iPhone.cor <- cor(iPhone.df)


cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}
# matrix of the p-value of the correlation
p.mat <- cor.mtest(iPhone.cor)

#Plot Correlation Matrix
plot.new(); dev.off()
corrplot(iPhone.cor, method="color", type="upper", tl.col="black", tl.srt=45, tl.cex = 0.7)
#Plot Correlation Matrix with significance as blank
corrplot(iPhone.cor, method="color", type="upper", p.mat = p.mat, sig.level = 0.01,insig = "blank", tl.col="black", tl.srt=45, tl.cex = 0.7)

#VARIANCE
iPhone.var <- nearZeroVar(iPhone.df, saveMetrics = TRUE)
iPhone.var
iPhone.nzv <- nearZeroVar(iPhone.df, saveMetrics = FALSE) 
iPhone.nzv

#New DF without Near Zero Variance attributes
iPhone.df.nzv <- iPhone.df[,-iPhone.nzv]
str(iPhone.df.nzv)


#RECURSIVE FEATURE ELIMINATION
# Let's sample the data before using RFE
set.seed(420)
iphoneSample <- iPhone.df[sample(1:nrow(iPhone.df), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults
rfeResults$optVariables

# Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
iPhone.df.rfe <- iPhone.df[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iPhone.df.rfe$iphonesentiment <- iPhone.df$iphonesentiment

# review outcome
str(iPhone.df.rfe)

#Change dependent variable to factor
iPhone.df$iphonesentiment <- as.factor(iPhone.df$iphonesentiment)
iPhone.df.nvz$iphonesentiment <- as.factor(iPhone.df.nvz$iphonesentiment)
iPhone.df.rfe$iphonesentiment <- as.factor(iPhone.df.rfe$iphonesentiment)


#MODELING -----
#Out Of The Box Function
OOTB <- function(Training, Testing, Variable){
  
  Model <- list()
  Predictions <- list()
  Metrics <- list()
  
  #C5.0
  c50Fit1 <- train(Variable ~ ., Training, method = "C5.0")
  c50Fit1.pred <- predict(c50Fit1, Testing)
  c50Fit1.table <- confusionMatrix(table(Testing$Variable, c50Fit1.pred$predictions))
  
  Model[["c5.0"]] <- c50Fit1
  Predictions[["c5.0"]] <- c50Fit1.pred
  Metrics[["c5.0"]] <- c50Fit1.table
  
  #RF
  set.seed(420)
  rfFit1 <- train(Variable ~ ., Training, method = "rf")
  rfFit1.pred <- predict(rfFit1, Testing)
  rfFit1.table <- confusionMatrix(table(Testing$Variable, rfFit1.pred$predictions))
  
  Model[["RF"]] <- rfFit1
  Predictions[["RF"]] <- rfFit1.pred
  Metrics[["RF"]] <- rfFit1.table
  
  #SVM
  set.seed(420)
  svmFit1 <- svm(Variable ~ ., Training)
  svmFit1.pred <- predict(svmFit1, Testing)
  svmFit1.table <- confusionMatrix(table(Testing$Variable, svmFit1.pred$predictions))
  
  Model[["SVM"]] <- svmFit1
  Predictions[["SVM"]] <- svmFit1.pred
  Metrics[["SVM"]] <- svmFit1.table
  
  #KNN
  set.seed(420)
  knnFit1 <- kknn(Variable ~ ., Training)
  knnFit1.pred <- predict(knnFit1, Testing)
  knnFit1.table <- confusionMatrix(table(Testing$Variable, knnFit1.pred$predictions))
  
  Model[["KNN"]] <- knnFit1
  Predictions[["KNN"]] <- knnFit1.pred
  Metrics[["KNN"]] <- knnFit1.table
  
  Output <- list(Model, Predictions, Metrics)
  Output
}
Results <- list()

#iPhone no feat. eng. 
#Data Partition
set.seed(420)
iPhone.df_trainIndex <- createDataPartition(iPhone.df$iphonesentiment, p = .7, list = F, times = 1)
iPhone.df_Train <- iPhone.df[ iPhone.df_trainIndex,]
iPhone.df_Test  <- iPhone.df[-iPhone.df_trainIndex,]

Results[["iPhone.df"]] <- OOTB(iPhone.df_Train, iPhone.df_Test, iphonesentiment)

#iPhone nzv. 
#Data Partition
set.seed(420)
iPhone.df.nzv_trainIndex <- createDataPartition(iPhone.df.nzv$iphonesentiment, p = .7, list = F, times = 1)
iPhone.df.nzv_Train <- iPhone.df.nzv[ iPhone.df.nzv_trainIndex,]
iPhone.df.nzv_Test  <- iPhone.df.nzv[-iPhone.df.nzv_trainIndex,]

Results[["iPhone.df.nzv"]] <- OOTB(iPhone.df.nzv_Train, iPhone.df.nzv_Test, iphonesentiment)

#iPhone nzv. 
#Data Partition
set.seed(420)
iPhone.df.rfe_trainIndex <- createDataPartition(iPhone.df.rfe$iphonesentiment, p = .7, list = F, times = 1)
iPhone.df.rfe_Train <- iPhone.df.rfe[ iPhone.df.rfe_trainIndex,]
iPhone.df.rfe_Test  <- iPhone.df.rfe[-iPhone.df.rfe_trainIndex,]
Results[["iPhone.df.rfe"]] <- OOTB(iPhone.df.rfe_Train, iPhone.df.rfe_Test, iphonesentiment)
