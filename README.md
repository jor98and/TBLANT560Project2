# TBLANT560Project2
Project two for TBLANT 560 at UWT for Dr. Sergio Davalos
---
title: "Project2"
author: "Jordan Anderson"
date: "3/16/2023"
output: html_document
---

```{r}
#install relelvant packages and set up knitr
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(warning = FALSE, message = FALSE)
knitr::opts_chunk$set(options(width = 1000))
library(e1071)
library(klaR)
library(nnet)
library(neuralnet)
library(MASS)
library(rpart)
library(randomForest)
install.packages("mlbench")
library(mlbench) 
library(caret)
library(stringr)
```

```{r}
#importdata
data("BreastCancer")

#Use all columns except for the ID and set the label to be the first one
bc <- cbind(BreastCancer[11],BreastCancer[2:10])

```

```{r}
#Look at the data to see its structure and if there are any missing values. 
summary(bc)
str(bc)
```

```{r}
#Impute missing values with mean
for (i in 1:ncol(bc)) {
  bc[is.na(bc[,i]), i] <- floor(mean(as.numeric(bc[,i]), na.rm = TRUE))
}
```

```{r}
#Make our label binary
bc$Malignant_1 <- ifelse(bc$Class == "malignant",1,0)

bc.num <- as.data.frame(apply(bc[,2:11],2,as.numeric))
bc.num <- cbind(bc.num[10],bc.num[1:9])
bc <- bc[,1:10] 
```

```{r}
#split data into tran and validation sets
#Sample 70% of our data for training
t_set <- sample(c(1:dim(bc)[1]), dim(bc)[1]*.7)
train.df <- bc[t_set, ]
valid.df <- bc[-t_set, ]
```

```{r}
#compare results with accuracy dataframe
accuracy.df <- data.frame(Model = seq(1,8,1), Train_Accuracy = rep(0,8) ,Valid_Accuracy = rep(0,8))
```

```{r}
#Support vector machines (SVM)
accuracy.df[1,1] <- "Support Vector Machines"
bc_svm <- svm(Class~.,train.df)
bc_svm.pred <- predict(bc_svm,train.df)
accuracy.df[1,2] <-confusionMatrix(as.factor(bc_svm.pred), as.factor(train.df$Class))$overall[1]
```

```{r}
#create predicition
bc_svm.v.pred <- predict(bc_svm, valid.df)
```

```{r}
#Add results to accuracy
accuracy.df[1,3] <- confusionMatrix(as.factor(bc_svm.v.pred), as.factor(valid.df$Class))$overall[1]
```

Naive Bayes
```{r}
#Naive Bayes
accuracy.df[2,1] <- "Naive Bayes"
bc_nb <-NaiveBayes(Class ~., train.df)
bc_nb.pred <- predict(bc_nb, train.df)
accuracy.df[2,2] <- confusionMatrix(as.factor(bc_nb.pred$class), as.factor(train.df$Class))$overall[1]
```

```{r}
#Create Predictions
bc_nb.v.pred <- predict(bc_nb, valid.df)
```

```{r}
#Add results to accuracy
accuracy.df[2,3] <- confusionMatrix(as.factor(bc_nb.v.pred$class), as.factor(valid.df$Class))$overall[1]
```


Neural Net
```{r}
#Setting up training and validation sets for Neural Net
train.num.df <- bc.num[t_set, ]
valid.num.df <- bc.num[-t_set, ]
```

```{r}
#Normalize data
norm_values <- preProcess(train.num.df[,2:10])
train.norm.df <- predict(norm_values, train.num.df)
valid.norm.df <- predict(norm_values, valid.num.df)
```

```{r}
#Create NN
accuracy.df[3,1] <- "Neural Net"
bc_nn <- neuralnet(Malignant_1 ~ .,linear.output = T, data = train.norm.df, hidden = c(2,4), rep = 5)
train.pred <- compute(bc_nn, train.norm.df)
train.class <- ifelse(train.pred$net.result > .5, 1, 0)
accuracy.df[3,2] <- confusionMatrix(as.factor(train.class), as.factor(train.num.df$Malignant_1))$overall[1]
```

```{r}
#create Predictions
valid.pred <- compute(bc_nn, valid.norm.df)
valid.class <- ifelse(valid.pred$net.result > .5, 1, 0)
```

```{r}
#Add results to accuracy
accuracy.df[3,3] <- confusionMatrix(as.factor(valid.class), as.factor(valid.norm.df$Malignant_1))$overall[1]
```


Decision Tree
```{r}
accuracy.df[4,1] <- "Decision Tree"
bc_tree <- rpart(Class~ ., train.df)
```

```{r}
#create predictions for DT
bc_tree.pred <- predict(bc_tree, train.df, type = "class")
bc_tree.v.pred <- predict(bc_tree, valid.df, type = "class")
```

```{r}
#Add results to accuracy
accuracy.df[4,2] <- confusionMatrix(as.factor(bc_tree.pred), as.factor(train.df$Class))$overall[1]
accuracy.df[4,3] <- confusionMatrix(bc_tree.v.pred, valid.df$Class)$overall[1]
```


Cross Validation
```{r}
accuracy.df[5,1] <- "Leave-1-Out CV"
ans <- numeric(length(as.numeric(valid.df[,1])))
for (i in 1:length(valid.df[,1])) {
  bc_tree2 <- rpart(Class ~ ., valid.df[-i,])
  bc_tree2.pred <- predict(bc_tree, valid.df[i,],type="class")
  ans[i] <- bc_tree2.pred
}
ans <- factor(ans,labels=levels(valid.df$Class))

accuracy.df[5,3] <- confusionMatrix(as.factor(ans), as.factor(valid.df$Class))$overall[1]
```


Regularised Descriminant Analysis (RDA)
```{r}
accuracy.df[6,1] <- "RDA"

bc_rda <- rda(Class ~ ., train.df)
bc_rda.pred <- predict(bc_rda, train.df)
accuracy.df[6,2] <- confusionMatrix(as.factor(bc_rda.pred$class), as.factor(train.df$Class))$overall[1]

bc_rda2 <- rda(Class ~ ., bc)
bc_rda.v.pred <- predict(bc_rda, valid.df)
accuracy.df[6,3] <- confusionMatrix(as.factor(bc_rda.v.pred$class), as.factor(valid.df$Class))$overall[1]
```


Random Forest
```{r}
accuracy.df[7,1] <- "Random Forests"

bc_rf <- randomForest(Class~., train.df, importance= TRUE)
bc_rf.pred <- predict(bc_rf, train.df)
accuracy.df[7,2] <- confusionMatrix(bc_rf.pred, train.df$Class)$overall[1]

bc_rf.v.pred <- predict(bc_rf, valid.df)
accuracy.df[7,3] <- confusionMatrix(as.factor(bc_rf.v.pred), as.factor(valid.df$Class))$overall[1]
```


Ensemble Model
```{r}
#Bring in all results from all models

ensamble.df <- cbind(as.data.frame(bc_svm.v.pred)[1],as.data.frame(bc_nb.v.pred)[1],as.data.frame(valid.class)[1],as.data.frame(bc_tree.v.pred)[1],as.data.frame(ans),as.data.frame(bc_rda.v.pred)[1],as.data.frame(bc_rf.v.pred)[1])
colnames(ensamble.df) <- c("svm", "nvb", "nnet","tree","LOOCV", "rda", "rf" )
```

```{r}
#Changing our label to numeric so we can add across our rows.  

ensamble.df$svm <- ifelse(ensamble.df$svm == "malignant",1,0)
ensamble.df$nvb <- ifelse(ensamble.df$nvb == "malignant",1,0)
ensamble.df$tree <- ifelse(ensamble.df$tree =="malignant",1,0)
ensamble.df$rda <- ifelse(ensamble.df$rda == "malignant",1,0)
ensamble.df$LOOCV <- ifelse(ensamble.df$LOOCV =="malignant",1,0)
ensamble.df$rf <- ifelse(ensamble.df$rf =="malignant",1,0)
```

```{r}
#Sum the rows and add it back to the df

sum_e <- as.matrix(ensamble.df)
ensamble.df$sum <- rowSums(sum_e)
```

```{r}
#Give us a class based answer again

ensamble.df$combo_class <- ifelse(ensamble.df$sum >= 4, "malignant", "benign")
```

```{r}
#Check our accuracy with our ensamble score

accuracy.df[8,1] <- "Combo Score"
accuracy.df[8,3] <- confusionMatrix(as.factor(ensamble.df$combo_class), as.factor(valid.df$Class))$overall[1]
```

```{r}
#Accuracy of all of the models
accuracy.df
```

