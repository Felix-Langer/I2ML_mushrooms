<<<<<<< HEAD
# I2ML Projekt #################################################################
# Thema: Mushroom Classification - Edible or Poisonous?
################################################################################

# Preparation ------------------------------------------------------------------
library(tidyverse)
library(mlr3verse)
library(ranger)

# path_project_directory = "~/Studium/Statistik/WiSe1920/Intro2ML/I2ML_mushrooms/" 
# path_project_directory = INSERTYOURPATHHERE
# path_vicky = "C:/Users/vszab/Desktop/Uni/Statistik/Wahlpflicht/Machine Learning/Projekt/"

# setwd(path_project_directory)
set.seed(123456)

mushrooms_data = read.csv("Data/mushrooms.csv") %>% 
  select(-veil.type) %>%  # veil.type has only 1 level => omit variable
  mutate(ID = row_number()) # for antijoin operation (test/train split)

str(mushrooms_data)

# Train Test Split
# Training set for tuning, test set for final evaluation on untouched data
train_test_ratio = .8
mushrooms_data_training = dplyr::sample_frac(tbl = mushrooms_data,
                                             size = train_test_ratio)
mushrooms_data_test = dplyr::anti_join(x = mushrooms_data,
                                       y = mushrooms_data_training,
                                       by = "ID")

mushrooms_data_training = dplyr::select(mushrooms_data_training, -ID)
mushrooms_data_ = dplyr::select(mushrooms_data_test, -ID) # Wieso wurde der gemacht?

# check domains of sampled variables
# make sure that every category of every variable is at least once in the sample
summary(mushrooms_data_training)

# Construct Classification Task ------------------------------------------------
task_mushrooms = TaskClassif$new(id = "mushrooms_data_training",
                               backend = mushrooms_data_training,
                               target = "class",
                               positive = "e") # "e" = edible
# Feature space:
task_mushrooms$feature_names
# Target variable:
# autoplot(task_mushrooms) + theme_bw()

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
  # for our falsely edible (although actually poisonous) classification
  msr("classif.sensitivity",
      id = "Sensitivity"),
  msr("classif.specificity",
      id = "Specificity"),
  msr("classif.ce", 
      id = "MMCE")
)

# Tuning -----------------------------------------------------------------------
# # what COULD we tune in the diffferent models?
# for(i in seq(1,5)){
#   print(learners[[i]]$param_set)
# }

# Setting Parameter for Autotune -----------------------------------------------
# Choose optimization algorithm:
# no need to etra randomize, try to go every step
tuner_grid_search = tnr("grid_search")  

measures_tuning = msr("classif.auc")
# Set the budget (when to terminate):
# we test every candidate
terminator_knn = term("evals", n_evals = 50)
terminator_mtry = term("evals", n_evals = 21)

# Autotune knn -----------------------------------------------------------------
# Define learner:
learner_knn = lrn("classif.kknn", predict_type = "prob")

# we want to tune k in knn:
learner_knn$param_set

# tune the chosen hyperparameters with these boundaries:
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
learner_ranger = lrn("classif.ranger", predict_type = "prob", importance = "impurity")

# we tune mtry for the random forest:
learner_ranger$param_set
# goal: see how close we get to the default mtry (floor(sqrt(p)))

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
  measures = measures_tuning, #autotoner nimmt trotzdem nur classif.ce her!?
  tune_ps = param_mtry, 
  terminator = terminator_mtry,
  tuner = tuner_grid_search
)

learner_tree = lrn("classif.rpart",
                   predict_type = "prob",
                   "cp" = 0.001) 
# set cp super low to enforce new splits so we get FPR < 1%

# Learner List------------------------------------------------------------------
learners = list(lrn("classif.featureless", predict_type = "prob"),
                lrn("classif.naive_bayes", predict_type = "prob"),
                learner_tree,
                lrn("classif.log_reg", predict_type = "prob"),
                tuner_ranger,
                tuner_knn
)
print(learners)

# Results ------------------------------------------------------------------------------------------
# only show warnings:
# lgr::get_logger("mlr3")$set_threshold("warn")

design = benchmark_grid(
  tasks = task_mushrooms,
  learners = learners,
  resamplings = resampling_outer_10CV
)
print(design)

execute_start_time <- Sys.time()
# Run the models (in 10 fold CV)
bmr = benchmark(design, store_models = TRUE) # takes about 15 minutes
evaluation_time <- Sys.time() - execute_start_time 
rm(execute_start_time)

print(bmr)

autoplot(bmr)
autoplot(bmr, type = "roc")

(tab = bmr$aggregate(measures))
bmr$score()

# Ranked Performance------------------------------------------------------------
ranks_performace = tab[, .(learner_id, rank_train = rank(-AUC), rank_test = rank(MMCE))] %>% 
  dplyr::arrange(rank_train)
print(ranks_performace)

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

# Cariable Importance ---------------------------------------------------------
# ranger
# learner_ranger = learners[[5]]
# Variable importance mode, one of 'none', 'impurity', 'impurity_corrected', 
# 'permutation'. The 'impurity' measure is the Gini index for classification, 
# the variance of the responses for regression and the sum of test statistics 
# (see splitrule) for survival.

filter = flt("importance", learner = learner_ranger)
filter$calculate(task_mushrooms)
feature_scores <- as.data.table(filter)

ggplot(data = feature_scores, aes(x = reorder(feature, -score), y = score)) +
  theme_bw() +
  geom_bar(stat = "identity") +
  ggtitle(label = "Variable Importance Mushroom Features") +
  labs(x = "Features", y = "Variable Importance Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_y_continuous(breaks = pretty(1:500, 10))


# Choose the best model and fit on whole dataset ---------------------------------------------------
tab
# classif.ranger.tuned or log_reg

# train tuner_ranger once again
# eigentlich sollte man als Terminator eine bestimmte perfomance oder stagnation in perf wählen
# da aber bei uns fast alle kombinationen 1 ergeben bei auc macht das wohl keinen Sinn
tuner_ranger = AutoTuner$new(
  learner = learner_ranger,
  resampling = resampling_inner_5CV,
  measures = measures,
  tune_ps = param_mtry, 
  terminator = terminator_mtry,
  tuner = tuner_grid_search
)
tuner_ranger$train(task_mushrooms)

# parameter
tuner_ranger$tuning_instance$archive(unnest = "params")[,c("mtry","AUC")]

tuner_ranger$tuning_result

# use those parameters for model
learner_final = lrn("classif.ranger",predict_type = "prob")
learner_final$param_set$values = tuner_ranger$tuning_result$params
# Train on whole train data
learner_final$train(task_mushrooms)
# Test on never touched test data (20% of whole data splitted at the beginning)
pred = learner_final$predict_newdata(newdata = mushrooms_data_test)
pred$score(measures)

# Tree Plot ------------------------------------------------------------------
# rpart CART implementation:
# rerun the model directly since we cannot access the rpart model in benchmark()
mod_rpart_tree <- rpart::rpart(class ~ ., 
                               data = mushrooms_data_training,
                               cp = 0.001)
summary(mod_rpart_tree)
mod_rpart_tree$splits
mod_rpart_tree$variable.importance

plot_tree_1 <- rattle::fancyRpartPlot(mod_rpart_tree,
                                      sub = "",
                                      caption = "CART Train Set 1",
                                      palettes = c("Blues",# edible
                                                   "Reds"))# poisonous

# rpart::plotcp(mod_rpart_tree) # pruning unnecessary

# Test prediction accuracy
t_pred = predict(mod_rpart_tree, mushrooms_data_test, type="class")
(confMat <- table(mushrooms_data_test$class, t_pred))
# sagt alle richtig voraus, aber könnte auch an den Daten liegen generalization error 
# bei der benchmark berechnung ist nicht ganz so gut wie bei anderen Modellen
tab

# Closing remarks -------------------------------------------------------------
## Logistic Regression convergence error: --------------------------------------
# Kudos: https://stats.stackexchange.com/questions/320661/unstable-logistic-regression-when-data-not-well-separated

# Perfect seperation will cause the optimization to not converge =>
# not converge will cause the coefficients to be very large =>
# the very large coefficients will cause "fitted probabilities numerically 0 or 1".
# This is exactly the case: Our separation is ridiculously good, hence "no convergence"
=======
>>>>>>> f6bdae5f05727f781702d1d2c49a223eec12b5fb
