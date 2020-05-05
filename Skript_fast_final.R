# I2ML Projekt #################################################################
# Thema: Mushroom Classification - Edible or Poisonous?
################################################################################

# Preparation ------------------------------------------------------------------
library(tidyverse) # for pipes and ggplot2
library(mlr3verse)
library(ranger) # random forests
library(rattle) # for CART plotting

# capture current ggplot theme and reset it at the end of the script:
original_ggplot_theme <- ggplot2::theme_get()
# ggplot theme for this script:
ggplot2::theme_set(ggplot2::theme_bw())

####################### @ CONNI ###########################
# 
# in theme_set solltest du die farben für alle plots setten können, dann
# musst du nicht scale_col_.. jedes mal ändern




# path_project_directory = INSERT_YOUR_PATH_HERE
# setwd(path_project_directory)

set.seed(123456)

mushrooms_data = read.csv("Data/mushrooms.csv") %>% 
  select(-veil.type)  # veil.type has only 1 level => omit variable

str(mushrooms_data)

# check domains of sampled variables
# make sure that every category of every variable is at least once in the sample
summary(mushrooms_data)

# Construct Classification Task ------------------------------------------------
task_mushrooms = TaskClassif$new(id = "mushrooms_data",
                               backend = mushrooms_data,
                               target = "class",
                               positive = "e") # "e" = edible
# Feature space:
task_mushrooms$feature_names
# Target variable:
# autoplot(task_mushrooms)

# Resampling Strategies ----------------------------------------------------------
# 5 fold cross validation for inner loop
resampling_inner_5CV = rsmp("cv", folds = 5L)
# 10 fold cross validation for outer loop
resampling_outer_10CV = rsmp("cv", folds = 10L)

# Performance Measures ---------------------------------------------------------
measures = list(
  msr("classif.auc",
      id = "AUC"),
  msr("classif.fpr",
      id = "False Positive Rate"), # false positive rate especially interesting
  # for our falsely edible - although actually poisonous - classifications
  msr("classif.sensitivity",
      id = "Sensitivity"),
  msr("classif.specificity",
      id = "Specificity"),
  msr("classif.ce", 
      id = "MMCE")
)

# Tuning -----------------------------------------------------------------------

# Setting Parameter for Autotune -----------------------------------------------
# Choose optimization algorithm:
# no need for etra randomization, try to go every step
tuner_grid_search = tnr("grid_search")

# evaluate performance on AUC:
measures_tuning = msr("classif.auc")
# Set the budget (=when to terminate):
# we test every candidate
terminator_knn = term("evals",
                      n_evals = 50)
terminator_mtry = term("evals",
                       n_evals = 21) # 21 possible features

# Autotune knn -----------------------------------------------------------------
# Define learner:
learner_knn = lrn("classif.kknn", predict_type = "prob")

# we want to tune k in knn:
learner_knn$param_set

# tune the chosen hyperparameter k with these boundaries:
param_k = ParamSet$new(
  list(
    ParamInt$new("k", lower = 1L, upper = 50)
  )
)

# Set up autotuner instance with the predefined setups
tuner_knn = AutoTuner$new(
  learner = learner_knn,
  resampling = resampling_inner_5CV,
  measures = measures_tuning, 
  tune_ps = param_k, 
  terminator = terminator_knn,
  tuner = tuner_grid_search
)

# execute nested resampling
# nested_resampling <- resample(task = task_mushrooms,
#                               learner = tuner_knn,
#                               resampling = resampling_outer_3CV)
# 
# nested_resampling$score()
# #nested_resampling$score() %>% unlist()
# nested_resampling$score()[, c("iteration", "classif.ce")]
# nested_resampling$aggregate()

# Autotune Random Forest ---------------------------------------------------------------------------
# Define learner:
learner_ranger = lrn("classif.ranger",
                     predict_type = "prob",
                     importance = "impurity") # Gini index (for classification)

# we tune mtry for the random forest:
learner_ranger$param_set
# goal: see how close we get to the default mtry (=floor(sqrt(p)))

# we will try all configurations: 1 to 21 features.
param_mtry = ParamSet$new(
  list(
    ParamInt$new("mtry", lower = 1L, upper = 21L)
  ) #sqrt(p) already quite good but how much of
  # an improvement is tuning?
)

# Set up autotuner instance with the predefined setups
tuner_ranger = AutoTuner$new(
  learner = learner_ranger,
  resampling = resampling_inner_5CV,
  measures = measures_tuning,
  tune_ps = param_mtry, 
  terminator = terminator_mtry,# pretty much all combinations yield perfect results
  # so we stick to evaluating 21 features as opposed to take e.g. stagnation as
  # termination criterion
  tuner = tuner_grid_search
)

# Learner List------------------------------------------------------------------
(learners = list(lrn("classif.featureless", predict_type = "prob"),
                lrn("classif.naive_bayes", predict_type = "prob"),
                lrn("classif.rpart", predict_type = "prob"),
                lrn("classif.log_reg", predict_type = "prob"),
                tuner_ranger,
                tuner_knn))

# Results ------------------------------------------------------------------------------------------
# Set how to run the learners:
(design = benchmark_grid(
  tasks = task_mushrooms,
  learners = learners, # learner list with 2 custom defined learners
  resamplings = resampling_outer_10CV)) # 10 fold crossvalidation on entire data set

# Run the models (in 10 fold CV) -----------------------------------------------

# lower the threshold to only show warnings:
lgr::get_logger("mlr3")$set_threshold("warn")
# TAKES 10-20 MINUTES #########################################################
bmr = benchmark(design, store_models = TRUE)
# reset console messages to default:
lgr::get_logger("mlr3")$set_threshold("info")

print(bmr)

# Compare Classification Errors among learners:
autoplot(bmr)
# ROC curve among learners:
autoplot(bmr, type = "roc") # very very good 

(tab_learner_performance = bmr$aggregate(measures))

# individual performances in inner loops:
bmr$score()

# Ranked Performance------------------------------------------------------------
learner_performance_ranked <- tab_learner_performance[,
                                                      .(learner_id,
                                                        rank_train = rank(-AUC),
                                                        rank_test = rank(MMCE))
                                                      ] %>% 
  arrange(rank_train) # sort in ascending order
learner_performance_ranked
# Logistic Regression and Random Forest clear winners

# Predictions knn
result_knn = tab$resample_result[[6]]
as.data.table(result_knn$prediction())

# Model Parameter
knn = bmr$score()[learner_id == "classif.kknn.tuned"]$learner
for (i in 1:10){
  print(knn[[i]]$tuning_result$params)
}

ranger = bmr$score()[learner_id == "classif.ranger.tuned"]$learner
for (i in 1:10){
  print(ranger[[i]]$tuning_result$params)
}

# Refit Winner Model on Entire Data Set----------------------------------------
learner_performance_ranked
tab_learner_performance
# Winner model choice: Random Forest since logistic regression is problematic
# when perfect separation is present (as is the case here)

# train tuner_ranger once again using the same specs as before
rm(tuner_ranger)
tuner_ranger = AutoTuner$new(
  learner = learner_ranger,
  resampling = resampling_inner_5CV,
  measures = measures,
  tune_ps = param_mtry, 
  terminator = term("none"), # pretty much all combinations yield perfect results
  # so we stick to evaluating 21 features as opposed to take e.g. stagnation as
  # termination criterion
  tuner = tuner_grid_search
)
tuner_ranger$tuning_instance

print(terminator_mtry)

# show only warnings:
# lgr::get_logger("mlr3")$set_threshold("warn")
tuner_ranger$train(task_mushrooms)
# reset console outputs to default:
lgr::get_logger("mlr3")$set_threshold("info")

# parameter
tuner_ranger$tuning_instance$archive(unnest = "params")[,c("mtry","AUC")]

tuner_ranger$tuning_result

# use those parameters for model
learner_final = lrn("classif.ranger",predict_type = "prob")
learner_final$param_set$values = tuner_ranger$tuning_result$params
# Train on whole train data
learner_final$train(task_mushrooms)


# Variable Importance Random Forest---------------------------------------------
# construct filter to extract variable importance in previously set up learner
filter_ranger = flt("importance", learner = learner_ranger)
# rerun learner on entire data set and store variable importance results
filter_ranger$calculate(task_mushrooms)

feature_scores <- as.data.table(filter_ranger)

ggplot(data = feature_scores, 
       aes(x = reorder(feature, -score),
           y = score)) +
  geom_bar(stat = "identity") +
  ggtitle(label = "Mushroom Features - Variable Importance in Random Forest") +
  labs(x = "Features", y = "Variable Importance Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_y_continuous(breaks = pretty(1:feature_scores$score[1],10))

# Tree Plot ------------------------------------------------------------------
# rpart CART implementation:
# rerun the model directly since we cannot access the rpart model in benchmark()
mod_rpart_tree <- rpart::rpart(class ~ ., 
                               data = mushrooms_data)
mod_rpart_tree
# summary(mod_rpart_tree)
mod_rpart_tree$splits
mod_rpart_tree$variable.importance

plot_tree_1 <- rattle::fancyRpartPlot(mod_rpart_tree,
                                      sub = "",
                                      caption = "CART Train Set 1",
                                      palettes = c("Blues",# edible
                                                   "Reds"))# poisonous

rpart::plotcp(mod_rpart_tree) # pruning unnecessary

# Test prediction accuracy
t_pred = predict(mod_rpart_tree, mushrooms_data_test, type="class")
(confMat <- table(mushrooms_data_test$class, t_pred))
# sagt alle richtig voraus, aber könnte auch an den Daten liegen generalization error 
# bei der benchmark berechnung ist nicht ganz so gut wie bei anderen Modellen
tab


# Reset ggplot theme -----------------------------------------------------------
theme_set(original_ggplot_theme)
# Closing remarks -------------------------------------------------------------
## Logistic Regression convergence error: --------------------------------------
# Kudos: https://stats.stackexchange.com/questions/320661/unstable-logistic-regression-when-data-not-well-separated

# Perfect seperation will cause the optimization to not converge =>
# not converge will cause the coefficients to be very large =>
# the very large coefficients will cause "fitted probabilities numerically 0 or 1".
# This is exactly the case: Our separation is ridiculously good, hence "no convergence"
