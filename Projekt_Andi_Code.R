# I2ML Projekt #####################################################################################
# Thema: Mushroom Classification - Edible or Poisonous?
####################################################################################################

# Preparation --------------------------------------------------------------------------------------
library(tidyverse)
library(mlr3verse)
library(ranger)

path_project_directory = "~/Studium/Statistik/WiSe1920/Intro2ML/I2ML_mushrooms/" 
# path_vicky = "C:/Users/vszab/Desktop/Uni/Statistik/Wahlpflicht/Machine Learning/Projekt/"

setwd(path_project_directory)
set.seed(123456)

mushrooms_data = read.csv("Data/mushrooms.csv") %>% 
  select(-veil.type) %>%  # veil.type has only 1 level => omit variable
  mutate(ID = row_number()) # for antijoin operation (test/train split)

str(mushrooms_data)

# Train Test Split
# Training set for tuning, test set for final evaluation
train_test_ratio = .8
mushrooms_data_training = dplyr::sample_frac(tbl = mushrooms_data,
                                              size = train_test_ratio)
mushrooms_data_test = dplyr::anti_join(x = mushrooms_data,
                                        y = mushrooms_data_training,
                                        by = "ID")

mushrooms_data_training = select(mushrooms_data_training, -ID)
mushrooms_data_ = select(mushrooms_data_test, -ID)

# check domains of sampled variables
# make sure that every category of every variable is at least once in the sample
summary(mushrooms_data_training)

# Construct Task --------------------------------------------------------------
task_shrooms = TaskClassif$new(id = "mushrooms_data_training",
                               backend = mushrooms_data_training,
                               target = "class",
                               positive = "e") # "e" = edible
# Feature space:
task_shrooms$feature_names
# Target variable:
autoplot(task_shrooms) + theme_bw()

# Resampling Strategy ---------------------------------------------------------
# resampling_inner_10CV = rsmp("cv", folds = 10L) # use this for final calculation
resampling_inner_3CV = rsmp("cv", folds = 3L)
# resampling_outer_10CV = rsmp("cv", folds = 10L) # use this for final calculation
resampling_outer_3CV = rsmp("cv", folds = 3L)

# Performance Measures ---------------------------------------------------------
measures = list(
  msr("classif.auc",
      id = "auc_train",
      predict_sets = "train"),
  msr("classif.auc",
      id = "auc_test"),
  msr("classif.fpr",
      id = "False Positive Rate"), # false positive rate especially interesting
  # for our falsely edible (although actually poisonous) classification
  msr("classif.sensitivity",
      id = "sensitivity"),
  msr("classif.specificity",
      id = "specificity"),
  #################################
  # warum nehmen wir eigtl nicht den CE?
  msr("classif.ce", 
      id = "class.ce")
  #################################
)

# Learner ------------------------------------------------------------------------------------------
learners = list(lrn("classif.featureless", predict_type = "prob", predict_sets = c("train", "test")),
                lrn("classif.naive_bayes", predict_type = "prob", predict_sets = c("train", "test")),
                lrn("classif.kknn", predict_type = "prob", predict_sets = c("train", "test")),
                lrn("classif.rpart", predict_type = "prob", predict_sets = c("train", "test")),
                lrn("classif.ranger", predict_type = "prob", predict_sets = c("train", "test")),
                lrn("classif.log_reg", predict_type = "prob", predict_sets = c("train", "test"))
)
print(learners)

####################################################################################################
# Tuning -------------------------------------------------------------------------------------------
# what COULD we tune in the diffferent models?
for(i in seq(1,5)){
  print(learners[[i]]$param_set)
}

# we want to tune k in knn:
learners[[3]]$param_set

# and mtry in the random forest:
learners[[5]]$param_set
# goal: see how close we get to the default mtry (floor(sqrt(p)))
# we will try all configurations: 1 to 21 features.

# tune the chosen hyperparameters with these boundaries:
param_k = ParamSet$new(
  list(
    ParamInt$new("k", lower = 1L, upper = 500)
  )
)

param_mtry = ParamSet$new(
  list(
    ParamInt$new("mtry", lower = 1L, upper = 21L)
  ) #sqrt(p) already quite good but how much of
  # an improvement is tuning?
)

# Choose optimization algorithm:
tuner_grid_search = tnr("grid_search") # try every step
# Performance Measure ----------------------------------------------------------
# measures = msr("classif.auc", id = "auc_train")
# measures
# Set the budget (when to terminate):
# we test every candidate
# terminator_all_candidates <- term("evals", n_evals = 500L)
terminator_abgespeckt <- term("evals", n_evals = 2)


# Set up autotuner instance with the predefined setups
shrooms_autotune = AutoTuner$new(
  learner = learners[[3]],
  resampling = resampling_inner_3CV,
  measures = msr("classif.auc",
                 id = "auc_train",
                 predict_sets = "train"), #autotoner nimmt trotzdem nur classif.ce her!?
  tune_ps = param_k, 
  terminator = terminator_abgespeckt,
  tuner = tuner_grid_search
  )

# execute nested resampling
nested_resampling <- resample(task = task_shrooms,
                              learner = shrooms_autotune,
                              resampling = resampling_outer_3CV)

nested_resampling$score()
nested_resampling$score() %>% unlist()


# ---------------------------
# ---------------------------

# Construct tuning instances:
tune_instance_knn = TuningInstance$new(
  task = task_shrooms,
  learner = learners[[3]],
  resampling = resampling_inner_3CV,
  measures = measures,
  param_set = param_k,
  terminator = terminator_abgespeckt
)
print(tune_instance_knn)

tune_instance_mtry = TuningInstance$new(
  task = task_shrooms,
  learner = learners[[5]],
  resampling = resampling_inner_3CV,
  measures = measures,
  param_set = param_mtry,
  terminator = terminator_abgespeckt
)
print(tune_instance_mtry)

# Start tuning:
result_tuning_k = tuner_grid_search$tune(tune_instance_knn)
result_tuning_mtry = tuner_grid_search$tune(tune_instance_mtry)
print(result_tuning_k)

# retrieve tuned hyperparameters:
tune_instance_knn$archive()
k_star = tune_instance_knn$archive(unnest = "params")[, c("k", "auc_train")]
mtry_star = tune_instance_mtry$archive(unnest = "params")[, c("mtry", "auc_train")]

# ---------------------------
# ---------------------------
# ---------------------------
# ---------------------------


# Results ------------------------------------------------------------------------------------------
design = benchmark_grid(
  tasks = task_shrooms,
  learners = learners_gewinner,
  resamplings = resampling_3CV
)
print(design)

# Run the models (in 3 fold CV)
bmr = benchmark(design)
print(bmr)

autoplot(bmr)
autoplot(bmr, type = "roc")

(tab = bmr$aggregate(measures))

learners[[3]]$predict_sets

# Ranked Performance
ranks = tab[, .(learner_id, rank_train = rank(-auc_train), rank_test = rank(-auc_test))]
print(ranks)

################################################################################
# Importance -------------------------------------------------------------------
# ranger
learner_i = lrn("classif.ranger", predict_type = "prob", importance = "impurity")
# Variable importance mode, one of 'none', 'impurity', 'impurity_corrected', 
# 'permutation'. The 'impurity' measure is the Gini index for classification, 
# the variance of the responses for regression and the sum of test statistics 
# (see splitrule) for survival.

filter = flt("importance", learner = learner_i)
filter$calculate(task)
as.data.table(filter)

# ggplot noch dazu, aus code demo für random forests!


## Logistic Regression convergence error: ------------
# Kudos: https://stats.stackexchange.com/questions/320661/unstable-logistic-regression-when-data-not-well-separated

# Perfect seperation will cause the optimization not converge, not converge will cause the coefficients to be very large,
# and the very large coefficient will cause "fitted probabilities numerically 0 or 1".
# This is exactly the case: Our separation is ridiculously good, hence "no convergence"

