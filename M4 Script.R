#INSTALL AND LOAD PACKAGES-----

#If pacman is missing we install it, then we load libraries
if (!require("pacman")) {
  install.packages("pacman")
} else{
  library(pacman)
  pacman::p_load(kknn, e1071, ggplot2, caret, readr, dplyr, tidyr, doParallel, plotly, corrplot)
}

#CLUSTERS -----
# Find how many cores are on your machine
detectCores() # 4

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
cl <- makeCluster(3)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # 3 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)


#UPLOAD DATA -----

iPhone.df <- read_csv("C:/Users/poni6/Desktop/Data Analysis/Modulo 4/Data/iphone_smallmatrix_labeled_8d.csv")
View(iPhone.df)
str(iPhone.df)
summary(iPhone.df)

Galaxy.df <- read_csv("C:/Users/poni6/Desktop/Data Analysis/Modulo 4/Data/galaxy_smallmatrix_labeled_9d.csv")
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




#FEATURE ENGINEERING (CORELATION)----

options(max.print = 1000000)
iPhone.cor <- cor(iPhone.df)
Galaxy.cor <- cor(Galaxy.df)

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
iPhone.p.mat <- cor.mtest(iPhone.cor)
Galaxy.p.mat <- cor.mtest(Galaxy.cor)

#Plot Correlation Matrix
plot.new(); dev.off()
corrplot(iPhone.cor, method="color", type="upper", tl.col="black", tl.srt=45, tl.cex = 0.7)
corrplot(Galaxy.cor, method="color", type="upper", tl.col="black", tl.srt=45, tl.cex = 0.7)

#Plot Correlation Matrix with significance as blank
corrplot(iPhone.cor, method="color", type="upper", p.mat = iPhone.p.mat, sig.level = 0.01,insig = "blank", tl.col="black", tl.srt=45, tl.cex = 0.7)
corrplot(Galaxy.cor, method="color", type="upper", p.mat = Galaxy.p.mat, sig.level = 0.01,insig = "blank", tl.col="black", tl.srt=45, tl.cex = 0.7)




#FEATURE ENGINEERING (VARIANCE) ------
iPhone.var <- nearZeroVar(iPhone.df, saveMetrics = TRUE)
iPhone.var
iPhone.nzv <- nearZeroVar(iPhone.df, saveMetrics = FALSE) 
iPhone.nzv

Galaxy.var <- nearZeroVar(Galaxy.df, saveMetrics = TRUE)
Galaxy.var
Galaxy.nzv <- nearZeroVar(Galaxy.df, saveMetrics = FALSE) 
Galaxy.nzv

#New DF without Near Zero Variance attributes
iPhone.df.nzv <- iPhone.df[,-iPhone.nzv]
str(iPhone.df.nzv)
Galaxy.df.nzv <- Galaxy.df[,-Galaxy.nzv]
str(Galaxy.df.nzv)

#FEATURE ENGINEERING (RFE) -------
# Let's sample the data before using RFE
set.seed(420)
iphoneSample <- iPhone.df[sample(1:nrow(iPhone.df), 1000, replace=FALSE),]
galaxySample <- Galaxy.df[sample(1:nrow(Galaxy.df), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
iphone.rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

galaxy.rfeResults <- rfe(galaxySample[,1:58], 
                         galaxySample$galaxysentiment, 
                         sizes=(1:58), 
                         rfeControl=ctrl)
# Get results
iphone.rfeResults
iphone.rfeResults$optVariables

galaxy.rfeResults
galaxy.rfeResults$optVariables

# Plot results
plot(iphone.rfeResults, type=c("g", "o"))
plot(galaxy.rfeResults, type=c("g", "o"))


# create new data set with rfe recommended features
iPhone.df.rfe <- iPhone.df[,predictors(iphone.rfeResults)]
Galaxy.df.rfe <- Galaxy.df[,predictors(galaxy.rfeResults)]

# add the dependent variable to iphoneRFE
iPhone.df.rfe$iphonesentiment <- iPhone.df$iphonesentiment
Galaxy.df.rfe$galaxysentiment <- Galaxy.df$galaxysentiment


# review outcome
str(iPhone.df.rfe)
str(Galaxy.df.rfe)

#Change dependent variable to factor
iPhone.df$iphonesentiment <- as.factor(iPhone.df$iphonesentiment)
iPhone.df.nzv$iphonesentiment <- as.factor(iPhone.df.nzv$iphonesentiment)
iPhone.df.rfe$iphonesentiment <- as.factor(iPhone.df.rfe$iphonesentiment)

Galaxy.df$galaxysentiment <- as.factor(Galaxy.df$galaxysentiment)
Galaxy.df.nzv$galaxysentiment <- as.factor(Galaxy.df.nzv$galaxysentiment)
Galaxy.df.rfe$galaxysentiment <- as.factor(Galaxy.df.rfe$galaxysentiment)

#MODELING FUNCTION -----
#Out Of The Box Function
OOTB.iPhone <- function(Training, Testing){
  
  Model <- list()
  Predictions <- list()
  Metrics <- list()
  
  #C5.0
  c50Fit1 <- train(iphonesentiment ~ ., Training, method = "C5.0")
  c50Fit1.pred <- predict(c50Fit1, Testing)
  c50Fit1.table <- confusionMatrix(table(Testing$iphonesentiment, c50Fit1.pred))
  
  Model[["c5.0"]] <- c50Fit1
  Predictions[["c5.0"]] <- c50Fit1.pred
  Metrics[["c5.0"]] <- c50Fit1.table
  
  #RF
  set.seed(420)
  rfFit1 <- train(iphonesentiment ~ ., Training, method = "rf")
  rfFit1.pred <- predict(rfFit1, Testing)
  rfFit1.table <- confusionMatrix(table(Testing$iphonesentiment, rfFit1.pred))
  
  Model[["RF"]] <- rfFit1
  Predictions[["RF"]] <- rfFit1.pred
  Metrics[["RF"]] <- rfFit1.table
  
  #SVM
  set.seed(420)
  svmFit1 <- svm(iphonesentiment ~ ., Training)
  svmFit1.pred <- predict(svmFit1, Testing)
  svmFit1.table <- confusionMatrix(table(Testing$iphonesentiment, svmFit1.pred))
  
  Model[["SVM"]] <- svmFit1
  Predictions[["SVM"]] <- svmFit1.pred
  Metrics[["SVM"]] <- svmFit1.table
  
  #KNN
  set.seed(420)
  knnFit1 <- kknn(iphonesentiment ~ ., Training, Testing)
  knnFit1.table <- confusionMatrix(table(Testing$iphonesentiment, knnFit1$fitted.values))
  
  Model[["KNN"]] <- knnFit1
  Metrics[["KNN"]] <- knnFit1.table
  Predictions[["KNN"]] <- knnFit1.table$overall
  
  Output <- list(Model, Predictions, Metrics)
  Output
}
OOTB.Galaxy <- function(Training, Testing){
  
  Model <- list()
  Predictions <- list()
  Metrics <- list()
  
  #C5.0
  c50Fit1 <- train(galaxysentiment ~ ., Training, method = "C5.0")
  c50Fit1.pred <- predict(c50Fit1, Testing)
  c50Fit1.table <- confusionMatrix(table(Testing$galaxysentiment, c50Fit1.pred))
  
  Model[["c5.0"]] <- c50Fit1
  Predictions[["c5.0"]] <- c50Fit1.pred
  Metrics[["c5.0"]] <- c50Fit1.table
  
  #RF
  set.seed(420)
  rfFit1 <- train(galaxysentiment ~ ., Training, method = "rf")
  rfFit1.pred <- predict(rfFit1, Testing)
  rfFit1.table <- confusionMatrix(table(Testing$galaxysentiment, rfFit1.pred))
  
  Model[["RF"]] <- rfFit1
  Predictions[["RF"]] <- rfFit1.pred
  Metrics[["RF"]] <- rfFit1.table
  
  #SVM
  set.seed(420)
  svmFit1 <- svm(galaxysentiment ~ ., Training)
  svmFit1.pred <- predict(svmFit1, Testing)
  svmFit1.table <- confusionMatrix(table(Testing$galaxysentiment, svmFit1.pred))
  
  Model[["SVM"]] <- svmFit1
  Predictions[["SVM"]] <- svmFit1.pred
  Metrics[["SVM"]] <- svmFit1.table
  
  #KNN
  set.seed(420)
  knnFit1 <- kknn(galaxysentiment ~ ., Training, Testing)
  knnFit1.table <- confusionMatrix(table(Testing$galaxysentiment, knnFit1$fitted.values))
  
  Model[["KNN"]] <- knnFit1
  Metrics[["KNN"]] <- knnFit1.table
  Predictions[["KNN"]] <- knnFit1.table$overall
  
  
  Output <- list(Model, Predictions, Metrics)
  Output
}
Results <- list()

#IPHONE MODELS -----
#iPhone no feat. eng. 
#Data Partition
set.seed(420)
iPhone.df_trainIndex <- createDataPartition(iPhone.df$iphonesentiment, p = .7, list = F, times = 1)
iPhone.df_Train <- iPhone.df[ iPhone.df_trainIndex,]
iPhone.df_Test  <- iPhone.df[-iPhone.df_trainIndex,]

Results[["iPhone.df"]] <- OOTB.iPhone(iPhone.df_Train, iPhone.df_Test)

#iPhone nzv. 
#Data Partition
set.seed(420)
iPhone.df.nzv_trainIndex <- createDataPartition(iPhone.df.nzv$iphonesentiment, p = .7, list = F, times = 1)
iPhone.df.nzv_Train <- iPhone.df.nzv[ iPhone.df.nzv_trainIndex,]
iPhone.df.nzv_Test  <- iPhone.df.nzv[-iPhone.df.nzv_trainIndex,]

Results[["iPhone.df.nzv"]] <- OOTB.iPhone(iPhone.df.nzv_Train, iPhone.df.nzv_Test)

#iPhone rfe. 
#Data Partition
set.seed(420)
iPhone.df.rfe_trainIndex <- createDataPartition(iPhone.df.rfe$iphonesentiment, p = .7, list = F, times = 1)
iPhone.df.rfe_Train <- iPhone.df.rfe[ iPhone.df.rfe_trainIndex,]
iPhone.df.rfe_Test  <- iPhone.df.rfe[-iPhone.df.rfe_trainIndex,]
Results[["iPhone.df.rfe"]] <- OOTB.iPhone(iPhone.df.rfe_Train, iPhone.df.rfe_Test)

#GALAXY MODELS -----
  
#Galaxy no feat. eng. 
#Data Partition
set.seed(420)
Galaxy.df_trainIndex <- createDataPartition(Galaxy.df$galaxysentiment, p = .7, list = F, times = 1)
Galaxy.df_Train <- Galaxy.df[ Galaxy.df_trainIndex,]
Galaxy.df_Test  <- Galaxy.df[-Galaxy.df_trainIndex,]

Results[["Galaxy.df"]] <- OOTB.Galaxy(Galaxy.df_Train, Galaxy.df_Test)

#Galaxy nzv. 
#Data Partition
set.seed(420)
Galaxy.df.nzv_trainIndex <- createDataPartition(Galaxy.df.nzv$galaxysentiment, p = .7, list = F, times = 1)
Galaxy.df.nzv_Train <- Galaxy.df.nzv[ Galaxy.df.nzv_trainIndex,]
Galaxy.df.nzv_Test  <- Galaxy.df.nzv[-Galaxy.df.nzv_trainIndex,]

Results[["Galaxy.df.nzv"]] <- OOTB.Galaxy(Galaxy.df.nzv_Train, Galaxy.df.nzv_Tes)

#Galaxy rfe. 
#Data Partition
set.seed(420)
Galaxy.df.rfe_trainIndex <- createDataPartition(Galaxy.df.rfe$galaxysentiment, p = .7, list = F, times = 1)
Galaxy.df.rfe_Train <- Galaxy.df.rfe[ Galaxy.df.rfe_trainIndex,]
Galaxy.df.rfe_Test  <- Galaxy.df.rfe[-Galaxy.df.rfe_trainIndex,]

Results[["Galaxy.df.rfe"]] <- OOTB.Galaxy(Galaxy.df.rfe_Train, Galaxy.df.rfe_Test)

#RECODING THE DEPENDANT VARIABLE -----


