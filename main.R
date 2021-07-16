
# Load required packages --------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(tidyverse)
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
library(corrplot)
if(!require(naniar)) install.packages("naniar", repos = "http://cran.us.r-project.org")
library(naniar)
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
library(grid)
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
library(RColorBrewer)
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
library(gridExtra)
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
library(matrixStats)

# Download data, wrangle it and save it -----------------------------------

## The original source of the data is
## https://archive.ics.uci.edu/ml/datasets/Ionosphere
## url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"


# download data
url <- "https://raw.githubusercontent.com/cgoldner/radar-ionosphere/main/data/ionosphere.data"
dest_file <- "data/ionosphere.data"
if(!exists("ionosphere")) download.file(url, destfile = dest_file)

# read data
ionosphere <- read.table("data/ionosphere.data", sep = ",")

# inspect data
str(ionosphere)
names(ionosphere)
nrow(ionosphere)

# rename columns
ionosphere <- setNames(ionosphere, c(paste0("x_",1:34), "y"))

# convert all predictor variables to numeric and convert outcome to factor
ionosphere <- ionosphere %>% mutate(x_1 = as.numeric(x_1),
                                    x_2 = as.numeric(x_2),
                                    y = factor(y, levels = c("g", "b")))

# check for NAs
vis_miss(ionosphere) # no NAs

# Create train and validation set -----------------------------------------

# split off validation set for later use
# split is train = 0.8 and validation = 0.2
set.seed(1, sample.kind="Rounding") # set seed to make splitting reproducible
val_index <- createDataPartition(ionosphere$y, times = 1, p = 0.2, list = FALSE)
validation <- ionosphere %>% slice(val_index)
training <- ionosphere %>% slice(-val_index)

save(training, file = "rda/training.rda")
save(validation, file = "rda/validation.rda")
save(ionosphere, file = "rda/ionosphere.rda")

# Data Exploration --------------------------------------------------------

# observe imaginary and real parts
# observe that first and second column have few values only
boxplot.matrix(as.matrix(training[, which(names(training) != "y")]))

# look at first columns impact on measurement type (good or bad)
hist_x_1 <- training %>% mutate(y = ifelse(y == "g", "good", "bad")) %>%
  ggplot(aes(x = x_1, fill = y)) +
  geom_histogram(aes(fill = y), binwidth = 0.05) +
  theme(legend.position="none") +
  labs(title = "Effect of x_1 column on measurement type") +
  theme(panel.grid.minor.x=element_blank(),
        panel.grid.major.x=element_blank())
hist_x_1

# compare proportion of measurement types
proportion_plot <- training %>% mutate(y = ifelse(y == "g", "good", "bad")) %>%
  group_by(y) %>%
  summarize(n = n()) %>%
  mutate(proportion = n/sum(n)) %>% ungroup() %>%
  mutate(group_var = "Measurement") %>%
  ggplot(aes(group_var, y = proportion, fill = y)) +
  theme_minimal() +
  geom_bar(position = "fill",stat = "identity") +
  theme(legend.position="bottom") +
  labs(fill = "Measurement type",
       x = "", 
       y = "Proportion", 
       title = "Proportion of good and bad measurements") +
  guides(y = "none") +
  geom_hline(yintercept = 0.6428571, size=2) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  coord_flip()
proportion_plot

grid.arrange(hist_x_1, proportion_plot, ncol = 1, heights = c(1, 0.5))

# compare all predictor variables
comparison_plot <- training %>%  mutate(y = ifelse(y == "g", "good", "bad")) %>%
  gather(pred_var, measure, -y) %>%
  mutate(pred_var_number = parse_number(pred_var)) %>%
  ggplot(aes(x = factor(pred_var_number), y = measure, fill = y)) +
  geom_boxplot() +
  labs(fill = "Measurement type",
       x = "Predictor variable", 
       y = "Measured value", 
       title = "Comparison of predictor variables") +
  theme(panel.grid.minor.y=element_blank(),
        panel.grid.major.y=element_blank())
comparison_plot

# inspect correlations between predictor variables
# for that remove x_1 and x_2
corrplot(cor(as.matrix(training[3:34])),
         type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)

# distinguish between real and imaginary parts
training_real <- training[seq(1, 33, 2)]
training_imaginary <- training[seq(2, 34, 2)]

corrplot(cor(as.matrix(training_real)),
         type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)
corrplot(cor(as.matrix(training_imaginary[-1])),
         type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)

training %>% mutate(y = ifelse(y == "g", 1, 0))
corrplot(cor(as.matrix((training %>% mutate(y = ifelse(y == "g", 1, 0)))[3:35])),
         type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)

# perform PCA to see if number of predictor variables can be reduced
pca_real <- prcomp(scale(training_real))
summary(pca_real)
pca_imaginary <- prcomp(scale(training_imaginary[-1])) # remove x_2 (because sd = 0)
summary(pca_imaginary)
pca <- prcomp(scale(training[1:34][-2])) # remove x_2 (because sd = 0)
summary(pca)

df_pc12 <- data.frame(pca$x[, 1:2], outcome = training$y) %>%
  mutate(pca = "all")
df_pc12_real <- data.frame(pca_real$x[, 1:2], outcome = training$y) %>%
  mutate(pca = "real")
df_pc12_imaginary <- data.frame(pca_imaginary$x[, 1:2], outcome = training$y) %>%
  mutate(pca = "imaginary")

# first two principal components do a bad job at distinguishing between good and bad measurements
pc12_plot <- df_pc12 %>% rbind(df_pc12_real) %>%
  rbind(df_pc12_imaginary) %>%
  mutate(pca = factor(pca, levels = c("all", "real", "imaginary"))) %>%
  mutate(outcome = ifelse(outcome == "g", "good", "bad")) %>%
  ggplot(aes(PC1,PC2, fill = outcome)) +
  geom_point(cex=2, pch=21, alpha = 0.7) +
  coord_fixed(ratio = 1) +
  facet_wrap( ~ pca) +
  labs(fill = "Measurement type")
pc12_plot

# more than half of predictor variables are required in order to explain at least 95% of variance
# hence there are not a few predictor variables that easily predict the outcome
pca_all_plot <- data.frame(cum_prop = summary(pca)$importance[3, ]) %>% 
  rownames_to_column("prcomp") %>%
  mutate(prcomp = parse_number(prcomp)) %>%
  ggplot(aes(x = prcomp, y = cum_prop)) +
  geom_point(size = 2, color = "burlywood4") +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  scale_x_continuous(breaks = seq(1, 34, 1)) +
  labs(x = "Principal component",
       y = "Cumulative proportion of explained variance",
       title = "PCA including real and imaginary parts") +
  geom_hline(yintercept = 0.95, color = "firebrick", linetype = "dashed")

pca_real_imaginary_plot <- data.frame(cum_prop = summary(pca_real)$importance[3, ]) %>% 
  rownames_to_column("prcomp") %>%
  mutate(prcomp = parse_number(prcomp)) %>%
  mutate(pca = "real") %>%
  rbind(
    data.frame(cum_prop = summary(pca_imaginary)$importance[3, ]) %>% 
    rownames_to_column("prcomp") %>%
    mutate(prcomp = parse_number(prcomp)) %>%
    mutate(pca = "imaginary")) %>%
  mutate(pca = factor(pca, levels = c("real", "imaginary"))) %>%
  ggplot(aes(x = prcomp, y = cum_prop, color = pca)) +
  geom_point(size = 2) +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  scale_x_continuous(breaks = seq(1, 17, 1)) +
  labs(x = "Principal component",
       y = "Cumulative proportion of explained variance",
       title = "Comparing PCA of real and imaginary parts") +
  geom_hline(yintercept = 0.95, color = "firebrick", linetype = "dashed") +
  scale_color_manual(values = c("deepskyblue4", "orangered"))

grid.arrange(pca_all_plot, pca_real_imaginary_plot, ncol = 1)


# Function to train models ------------------------------------------------


# REGRESSION:
# lm + cutoff
# glm (logit) + cutoff
# knn + cutoff
# 
# CLASSIFICATION:
# lda
# qda
# rf
# svm
# 
# ENSEMBLE:
# use majority vote on odd number of models
# 
# GENERAL:
# for 1 model (use x_1 to deduce bad measurement, exclude x_2):
# 1. split into "test" and "train" set.
# 2. optimize model parameters on train set.
# 3. evaluate optimized model on test set (compute accuracy).
# 4. repeat 1-3 for example 5 times, take average accuracy and average optimization parameters.
# 
# 5. apply 1-4 for all models.
# 6. ensemble best models into one final model.
# 7. evaluate final model on final hold out set "validation".


train_models <- function(s,
                         final_training = FALSE,
                         models = c("lm", "glm", "lda", "qda", "knn", "rf", "svm"),
                         tune_grid_rf = data.frame(mtry = seq(1, 10, 1)),
                         tune_grid_svm = expand.grid(scale = c(0, .1, 0.01),
                                                     C = c(0.1, 1, 0.10),
                                                     degree = c(1, 2, 1)),
                         tune_grid_knn = data.frame(k = seq(1, 20, 1))) {
  
    # define train and test set
    if(final_training == FALSE){
      # split off test set for evaluation of model performance
      # split is train = 0.8 and test = 0.2
      set.seed(s, sample.kind="Rounding") # set seed to run function several times with different seeds
      # this simulates cross-validation later on
      test_index <- createDataPartition(training$y, times = 1, p = 0.2, list = FALSE)
      test <- training %>% slice(test_index)
      train <- training %>% slice(-test_index)
    }
    
    if(final_training == TRUE){
      test <- validation
      train <- training
    }
    
    # initialize lists and data frames for saving and returning models and outputs
    model_list <- list()
    model_infos <- list()
    variable_importance <- data.frame(variable = paste0("x_",1:34))
    accuracy <- data.frame()
    predictions <- data.frame(matrix(ncol = nrow(test), nrow = 0))
    names(predictions) <- 1:nrow(test)


  
  # prepare train and test sets
  train_x <- train %>% select(-"x_1", -"x_2", -"y")
  
  train_y_factor <- train %>% pull("y")
  train_y_numeric <- train %>% mutate(y = ifelse(y == "g", 1, 0)) %>% pull("y")
  
  test_factor <- test
  test_numeric <- test %>% mutate(y = ifelse(y == "g", 1, 0))
  
  if("lm" %in% models){
    # lm + cutoff
    fit_lm <- train(train_x,
                    train_y_numeric,
                    method = "lm")
    pred_prop <- predict(fit_lm,
                         newdata = test_numeric)
    pred_lm <- ifelse(pred_prop >= 0.5, "g", "b")
    pred_lm[test_numeric$x_1 == 0] <- "b"
    pred_lm <- pred_lm %>% factor(levels = c("g", "b"))
    acc_lm <- mean(pred_lm == test_factor$y)
    # prepare saving lm results
    var_imp_tmp <- varImp(fit_lm$finalModel)
    names(var_imp_tmp) <- "lm"
    var_imp_tmp <- var_imp_tmp %>%
      mutate(variable = rownames(.))
    # save lm results
    variable_importance <- variable_importance %>%
      left_join(var_imp_tmp, by = "variable")
    accuracy <- accuracy %>%
      rbind(data.frame(model_name = "lm", acc = acc_lm))
    predictions <- predictions %>%
      rbind(data.frame(t(data.frame(pred_lm))))
    model_infos[["lm"]] <- list("lm", fit_lm$finalModel, fit_lm$bestTune)
  }
  
  if("glm" %in% models){
    # glm + cutoff
    fit_glm <- train(train_x,
                     train_y_numeric,
                     method = "glm")
    pred_prop <- predict(fit_glm,
                         newdata = test_numeric)
    pred_glm <- ifelse(pred_prop >= 0.5, "g", "b")
    pred_glm[test_numeric$x_1 == 0] <- "b"
    pred_glm <- pred_glm %>% factor(levels = c("g", "b"))
    acc_glm <- mean(pred_glm == test_factor$y)
    # prepare saving glm results
    var_imp_tmp <- varImp(fit_glm$finalModel)
    names(var_imp_tmp) <- "glm"
    var_imp_tmp <- var_imp_tmp %>%
      mutate(variable = rownames(.))
    # save glm results
    variable_importance <- variable_importance %>%
      left_join(var_imp_tmp, by = "variable")
    accuracy <- accuracy %>%
      rbind(data.frame(model_name = "glm", acc = acc_glm))
    predictions <- predictions %>%
      rbind(data.frame(t(data.frame(pred_glm))))
    model_infos[["glm"]] <- list("glm", fit_glm$finalModel, fit_glm$bestTune)
  }
  
  if("knn" %in% models){
    # knn + cutoff
    fit_knn <- train(train_x,
                     train_y_numeric,
                     method = "knn",
                     tuneGrid = tune_grid_knn)
    pred_prop <- predict(fit_knn$finalModel,
                         newdata = test_numeric  %>%
                           select(-"x_1", -"x_2", -"y"))
    pred_knn <- ifelse(pred_prop >= 0.5, "g", "b")
    pred_knn[test_numeric$x_1 == 0] <- "b"
    pred_knn <- pred_knn %>% factor(levels = c("g", "b"))
    acc_knn <- mean(pred_knn == test_factor$y)
    # save knn results
    accuracy <- accuracy %>%
      rbind(data.frame(model_name = "knn", acc = acc_knn))
    predictions <- predictions %>%
      rbind(data.frame(t(data.frame(pred_knn))))
    model_infos[["knn"]] <- list("knn", fit_knn$finalModel, fit_knn$bestTune)
  }

  if("lda" %in% models){
    # lda
    fit_lda <- train(train_x,
                     train_y_factor,
                     method = "lda")
    pred_prop <- predict(fit_lda$finalModel,
                         newdata = test_factor  %>%
                           select(-"x_1", -"x_2", -"y"))
    pred_lda <- pred_prop$class
    pred_lda[test_numeric$x_1 == 0] <- "b"
    pred_lda <- pred_lda %>% factor(levels = c("g", "b"))
    acc_lda <- mean(pred_lda == test_factor$y)
    # save lda results
    accuracy <- accuracy %>%
      rbind(data.frame(model_name = "lda", acc = acc_lda))
    predictions <- predictions %>%
      rbind(data.frame(t(data.frame(pred_lda))))
    model_infos[["lda"]] <- list("lda", fit_lda$finalModel, fit_lda$bestTune)
  }
  
  if("qda" %in% models){
    # qda
    fit_qda <- train(train_x,
                     train_y_factor,
                     method = "qda")
    pred_prop <- predict(fit_qda$finalModel,
                         newdata = test_factor  %>%
                           select(-"x_1", -"x_2", -"y"))
    pred_qda <- pred_prop$class
    pred_qda[test_numeric$x_1 == 0] <- "b"
    pred_qda <- pred_qda %>% factor(levels = c("g", "b"))
    acc_qda <- mean(pred_qda == test_factor$y)
    # save qda results
    accuracy <- accuracy %>%
      rbind(data.frame(model_name = "qda", acc = acc_qda))
    predictions <- predictions %>%
      rbind(data.frame(t(data.frame(pred_qda))))
    model_infos[["qda"]] <- list("qda", fit_qda$finalModel, fit_qda$bestTune)
  }
   
  if("rf" %in% models){   
    # rf
    # do not scale data since input data is already between -1 and 1
    # we assume input data has gone through some kind of scaling already
    # moreover, scaling with scale(train_x) yields worse results
    # do not exclude x_1 since rf incorporates x_1 automatically
    fit_rf <- train(train %>% select(-"x_2", -"y"),
                    train_y_factor,
                    method = "rf",
                    metric = "Accuracy",
                    tuneGrid = tune_grid_rf)
    pred_rf <- predict(fit_rf$finalModel,
                       newdata = test_factor  %>%
                         select(-"x_2", -"y"))
    pred_rf <- pred_rf %>% factor(levels = c("g", "b"))
    acc_rf <- mean(pred_rf == test_factor$y)
    # prepare saving rf results
    var_imp_tmp <- varImp(fit_rf$finalModel)
    names(var_imp_tmp) <- "rf"
    var_imp_tmp <- var_imp_tmp %>%
      mutate(variable = rownames(.))
    # save rf results
    variable_importance <- variable_importance %>%
      left_join(var_imp_tmp, by = "variable")
    accuracy <- accuracy %>%
      rbind(data.frame(model_name = "rf", acc = acc_rf))
    predictions <- predictions %>%
      rbind(data.frame(t(data.frame(pred_rf))))
    model_infos[["rf"]] <- list("rf", fit_rf$finalModel, fit_rf$bestTune)
  }
  
  if("svm" %in% models){
    # svm
    # do not scale data since input data is already between -1 and 1
    # we assume input data has gone through some kind of scaling already
    # moreover, scaling with scale(train_x) yields worse results
    # do not exclude x_1 since SVM incorporates x_1 automatically
    fit_svm <- train(train %>% select(-"x_2", -"y"),
                 train_y_factor,
                 method = "svmPoly",
                 tuneLength = 4, 
                 tuneGrid = tune_grid_svm)
    pred_svm <- predict(fit_svm,
                       newdata = test_numeric  %>%
                         select(-"x_2", -"y"))
    pred_svm <- pred_svm %>% factor(levels = c("g", "b"))
    acc_svm <- mean(pred_svm == test_factor$y)
    # prepare saving svm results
    var_imp_tmp <- varImp(fit_svm)$importance["g"]
    names(var_imp_tmp) <- "svm"
    var_imp_tmp <- var_imp_tmp %>%
      mutate(variable = rownames(.))
    # save svm results
    variable_importance <- variable_importance %>%
      left_join(var_imp_tmp, by = "variable")
    accuracy <- accuracy %>%
      rbind(data.frame(model_name = "svm", acc = acc_svm))
    predictions <- predictions %>%
      rbind(data.frame(t(data.frame(pred_svm))))
    model_infos[["svm"]] <- list("svm", fit_svm, fit_svm$bestTune)
  }  

  # process saved results
  predictions <- predictions %>% 
    unname() %>%
    t() %>%
    data.frame() %>%
    lapply(as.factor) %>%
    data.frame() %>%
    cbind(y = test_factor$y)
  rownames(variable_importance) <- variable_importance$variable
  variable_importance <- variable_importance %>%
    select(-variable) %>% 
    t() %>%
    as.data.frame()
  
  # pack all results into a list and return it
  model_list <- list(model_infos, variable_importance, accuracy, predictions)
  names(model_list) <- c("model_infos", "variable_importance", "accuracy", "predictions")
  
  # the output of the train_models function is "model_list":
  # model_list = list over all trained models:
  # list(list(model_infos), df(variable_importance), df(accuracy), df(predictions))
  # with model_infos = list("model_name", "tuned_model", "tuning_parameters")
  
  model_list
}


# Run all models on train set to tune their parameters --------------------
#
# run k times with random test sets
# then choose the best models
# model_list_train is a list of length k
# each entry is a the output of one run of the train_models function

k <- 10
model_list_train <- list()

for (s in 1:k) {
  model_list_train[[s]] <- train_models(s, final_training = FALSE)
}

View(model_list_train)


# Evaluate Accuracy to choose best models ---------------------------------
#
# choices of models: rf > svm > lda
# other models performed the same or worse

accuracy_train_df <- data.frame(model_name = model_list_train[[1]]$accuracy$model_name)

for (s in 1:k) {
  accuracy_train_df <- accuracy_train_df %>%
    left_join(model_list_train[[s]]$accuracy, by = "model_name")
  accuracy_train_df <- accuracy_train_df %>% rename_at("acc", list(~paste0("acc_run_", s)))
}

table_train_acc <- accuracy_train_df %>%
  mutate(acc_average = rowMeans(accuracy_train_df[, colnames(accuracy_train_df) != "model_name"])) %>%
  arrange(desc(acc_average))
View(table_train_acc)


# Check if lm, glm, lda yield the same results ----------------------------
# if yes, then do not use lm and glm, but only use lda

for (s in 1:k) {
  print(identical(model_list_train[[s]]$predictions$pred_lm,
                  model_list_train[[s]]$predictions$pred_lda))
}

for (s in 1:k) {
  print(identical(model_list_train[[s]]$predictions$pred_glm,
                  model_list_train[[s]]$predictions$pred_lda))
}



# Evaluate tuning parameters ----------------------------------------------

# tuning parameter mtry for rf
rf_mtry <- data.frame()

for (s in 1:k) {
  rf_mtry <- rf_mtry %>% rbind(model_list_train[[s]]$model_infos$rf[[3]])
}

rf_mtry <- rf_mtry %>% summarize(mtry = round(mean(mtry)))

# for svm the tuned parameters are almost constant
# notice that if we allow higher degrees to be used for tuning in svm
# then accuracy of svm suffers from overfitting:
# train_acc degree = 3: 0.8946429
# train_acc degree = 2: 0.9107143 -> d=2 is best degree
# train_acc degree = 1: 0.8803571
# in general svm should not perform worse than linear methods
# since it generalizes linear approaches
for (s in 1:k) {
  print(model_list_train[[s]]$model_infos$svm[[3]])
}

svm_tune <- model_list_train[[s]]$model_infos$svm[[3]]


# Function to train ensembled models --------------------------------------
#
# ensemble rf, svm, lda

train_ensemble <- function(s, final_training = FALSE) {
  ensemble_list <- train_models(s, final_training,
                                models = c("rf", "svm", "lda"),
                                tune_grid_rf = rf_mtry,
                                tune_grid_svm = svm_tune)
  pred_ensemble <- ensemble_list$predictions %>%
    mutate(votes_good = ifelse(pred_lda == "g", 1, 0)) %>%
    mutate(votes_good = ifelse(pred_rf == "g", votes_good + 1, votes_good)) %>% 
    mutate(votes_good = ifelse(pred_svm == "g", votes_good + 1, votes_good)) %>%
    mutate(pred_ensemble = ifelse(votes_good >= 2, "g", "b")) %>%
    mutate(pred_ensemble = factor(pred_ensemble, levels = c("g", "b"))) %>% 
    select(-"votes_good")
}


# Evaluate accuracy of ensemble -------------------------------------------

k <- 10
pred_ensemble_list <- list()

for (s in 1:k) {
  pred_ensemble_list[[s]] <- train_ensemble(s, final_training = FALSE)
}

accuracy_ensemble_df <- data.frame()
for (s in 1:k) {
  accuracy_ensemble_df <- rbind(accuracy_ensemble_df, 
                                mean(pred_ensemble_list[[s]]$pred_ensemble == pred_ensemble_list[[s]]$y))
}
names(accuracy_ensemble_df) <- "acc_ensemble_per_run"
acc_ensemble <- accuracy_ensemble_df %>%
  summarize(mean(acc_ensemble_per_run))



# Compare ensemble to other methods ---------------------------------------
#
# conclude that ensemble does not perform much better
# thus choose the simpler model, namely rf

acc_comparison <- table_train_acc %>%
    filter(model_name %in% c("rf", "svm", "lda")) %>%
    select(model_name, acc_average) %>%
    t() %>%
    cbind(c("ensemble", acc_ensemble[1,1])) %>%
    data.frame() %>%
    unname()
names(acc_comparison) <- acc_comparison["model_name",]
acc_comparison[2, ] %>%
  mutate(ensemble = round(as.numeric(ensemble), digits = 7))


# Compare types of errors of rf and ensemble ------------------------------
# ensemble misclassifies "bad" as "good" more often than rf does
# i.e. lower specificity

model_ensemble <- train_ensemble(s = 1, final_training = FALSE)

confusionMatrix(model_ensemble$pred_rf, model_ensemble$y)
confusionMatrix(model_ensemble$pred_ensemble, model_ensemble$y)



# Final model -------------------------------------------------------------

final_model <- train_models(s = 1, final_training = TRUE,
             models = c("rf"),
             tune_grid_rf = rf_mtry)



# Evaluate overall performance of final model -----------------------------

# final model still suffers from misclassifying "bad" as "good"
# i.e. lower specificity than sensitivity if positive class is "good"
confusionMatrix(final_model$predictions$pred_rf, final_model$predictions$y)

acc_final_model <- confusionMatrix(final_model$predictions$pred_rf, final_model$predictions$y)$overall["Accuracy"]
acc_final_model

F_meas(data = factor(final_model$predictions$pred_rf, levels = c("g", "b")), reference = final_model$predictions$y)



# Evaluate variable importance in final model -----------------------------

evaluation_variable_importance <- final_model$variable_importance %>%
  select(-"x_2") %>%
  gather(variable, importance) %>%
  mutate(variable = parse_number(variable)) %>%
  mutate(real = ifelse(variable %% 2 == 0 , "imaginary", "real")) %>% # modulo division by 2
  mutate(real = factor(real, levels = c("real", "imaginary")))
avg_importance_real_imaginary <- evaluation_variable_importance %>% 
  group_by(real) %>% 
  summarize(avg = mean(importance))
avg_importance <- evaluation_variable_importance %>%
  summarize(avg = mean(importance))

# observe that few variables are very important
variable_importance_plot <- evaluation_variable_importance %>% 
  ggplot(aes(x = factor(variable), y = importance, fill = real)) +
  theme(legend.position="top") + 
  geom_col() +
  scale_fill_manual(values = c("deepskyblue4", "orangered")) +
  geom_hline(yintercept = filter(avg_importance_real_imaginary, real == "real")$avg,
             color = "deepskyblue4") +
  geom_hline(yintercept = filter(avg_importance_real_imaginary, real == "imaginary")$avg,
             color = "orangered") +
  geom_hline(yintercept = avg_importance$avg,
             color = "black") +
  labs(x = "Predictor variable",
       y = "Importance",
       title = "Variable importance in final model", 
       fill = "")
variable_importance_plot


# Evaluate errors of final model ------------------------------------------

# claim: misclassified observations are those whose real and imaginary parts of the pulses are very unusual
# or at least unusual in the most important variables
# answer: this is only partly true

unlikely_values_and_misclassification <- scale(ionosphere %>% select(-"y", -"x_2")) %>%
  data.frame() %>%
  slice(val_index) %>%
  cbind(final_model$predictions) %>%
  filter(pred_rf != y) %>% # filter misclassified
  mutate(error_type = factor(ifelse(y == "b", "FP", "FN"))) %>% 
  mutate(observation_number = 1:length(error_type)) %>%
  select(-"pred_rf", -"y") %>% 
  gather(variable, measure, -"error_type", -"observation_number") %>% 
  mutate(variable = parse_number(variable)) %>%
  mutate(observation_number = paste0("Misclassification ", observation_number)) %>% 
  ggplot(aes(x = factor(variable), y = measure, color=error_type)) +
  theme(panel.grid.minor.y=element_blank(),
        panel.grid.major.y=element_blank(),
        legend.position="bottom") +
  geom_point() +
  scale_y_continuous(limits = c(-3, 3)) +
  geom_hline(yintercept = 2, color = "firebrick", linetype = "dashed") +
  geom_hline(yintercept = -2, color = "firebrick", linetype = "dashed") +
  facet_wrap(observation_number ~ ., ncol = 1) +
  labs(fill = "Error type",
       x = "Predictor variable", 
       y = "z-score of measured value") +
  scale_color_manual(values = c("darkorchid2", "navy"))
unlikely_values_and_misclassification


grid.arrange(variable_importance_plot, unlikely_values_and_misclassification, ncol = 1, heights = c(4, 10))









