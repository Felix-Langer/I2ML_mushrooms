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

# Construct Task -----------------------------------------------------------------------------------
task_shrooms = TaskClassif$new(id = "mushrooms_data_training",
                               backend = mushrooms_data_training,
                               target = "class",
                               positive = "e") # "e" = edible
# Feature space:
task_shrooms$feature_names
# Target variable:
autoplot(task_shrooms) + theme_bw()

# Resampling Strategy ------------------------------------------------------------------------------
# resampling_inner_10CV = rsmp("cv", folds = 10L) # use this for final calculation
resampling_inner_3CV = rsmp("cv", folds = 3L)
# resampling_outer_10CV = rsmp("cv", folds = 10L) # use this for final calculation
resampling_outer_3CV = rsmp("cv", folds = 3L)

# Performance Measures -----------------------------------------------------------------------------
measures = list(
  msr("classif.auc",
      id = "auc"),
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
      id = "mmce")
  #################################
)

####################################################################################################
# Tuning -------------------------------------------------------------------------------------------
# # what COULD we tune in the diffferent models?
# for(i in seq(1,5)){
#   print(learners[[i]]$param_set)
# }

# Setting Parameter for Autotune -------------------------------------------------------------------
# Choose optimization algorithm:
tuner_grid_search = tnr("grid_search") # try every step

measures_tuning = msr("classif.auc")

# Set the budget (when to terminate):
# we test every candidate
# terminator_all_candidates <- term("evals", n_evals = 500L)
terminator_abgespeckt <- term("evals", n_evals = 2)


# Autotune knn -------------------------------------------------------------------------------------
learner_knn = lrn("classif.kknn", predict_type = "prob")

# we want to tune k in knn:
learner_knn$param_set

# tune the chosen hyperparameters with these boundaries:
param_k = ParamSet$new(
  list(
    ParamInt$new("k", lower = 1L, upper = 100)
  )
)

# Set up autotuner instance with the predefined setups
tuner_knn = AutoTuner$new(
  learner = learner_knn,
  resampling = resampling_inner_3CV,
  measures = measures, #autotoner nimmt trotzdem nur classif.ce her!?
  tune_ps = param_k, 
  terminator = terminator_abgespeckt,
  tuner = tuner_grid_search
  )

# execute nested resampling
# nested_resampling <- resample(task = task_shrooms,
#                               learner = tuner_knn,
#                               resampling = resampling_outer_3CV)
# 
# nested_resampling$score()
# #nested_resampling$score() %>% unlist()
# nested_resampling$score()[, c("iteration", "classif.ce")]
# nested_resampling$aggregate()

# Autotune Random Forest ---------------------------------------------------------------------------
learner_ranger = lrn("classif.ranger", predict_type = "prob", importance = "impurity")

# and mtry in the random forest:
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
  resampling = resampling_inner_3CV,
  measures = measures, #autotoner nimmt trotzdem nur classif.ce her!?
  tune_ps = param_mtry, 
  terminator = terminator_abgespeckt,
  tuner = tuner_grid_search
)

# Learner ------------------------------------------------------------------------------------------
learners = list(lrn("classif.featureless", predict_type = "prob"),
                lrn("classif.naive_bayes", predict_type = "prob"),
                lrn("classif.rpart", predict_type = "prob"),
                lrn("classif.log_reg", predict_type = "prob"),
                tuner_ranger,
                tuner_knn
)
print(learners)

# Results ------------------------------------------------------------------------------------------
# only show warnings:
# lgr::get_logger("mlr3")$set_threshold("warn")

design = benchmark_grid(
  tasks = task_shrooms,
  learners = learners,
  resamplings = resampling_outer_3CV
)
print(design)

# Run the models (in 3 fold CV)
bmr = benchmark(design, store_models = TRUE)
print(bmr)

autoplot(bmr)
autoplot(bmr, type = "roc")

(tab = bmr$aggregate(measures))

# Ranked Performance
ranks = tab[, .(learner_id, rank_train = rank(-auc), rank_test = rank(mmce))]
print(ranks)

# predictions
result_knn = tab$resample_result[[6]]
as.data.table(result_knn$prediction())

# models
knn = bmr$score()[learner_id == "classif.kknn.tuned"]$learner
for (i in 1:3){
  print(knn[[i]]$tuning_result$params)
}

ranger = bmr$score()[learner_id == "classif.ranger.tuned"]$learner
for (i in 1:3){
  print(ranger[[i]]$tuning_result$params)
}

# Importance ---------------------------------------------------------------------------------------
# ranger
# learner_ranger = learners[[5]]
# Variable importance mode, one of 'none', 'impurity', 'impurity_corrected', 
# 'permutation'. The 'impurity' measure is the Gini index for classification, 
# the variance of the responses for regression and the sum of test statistics 
# (see splitrule) for survival.

filter = flt("importance", learner = learner_ranger)
filter$calculate(task_shrooms)
feature_scores <- as.data.table(filter)

ggplot(data = feature_scores, aes(x = reorder(feature, -score), y = score)) +
  theme_bw() +
  geom_bar(stat = "identity") +
  ggtitle(label = "Variable Importance Mushroom Features") +
  labs(x = "Features", y = "Variable Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_y_continuous(breaks = pretty(1:550, 15))


## Logistic Regression convergence error: ----------------------------------------------------------
# Kudos: https://stats.stackexchange.com/questions/320661/unstable-logistic-regression-when-data-not-well-separated

# Perfect seperation will cause the optimization not converge, not converge will cause the coefficients to be very large,
# and the very large coefficient will cause "fitted probabilities numerically 0 or 1".
# This is exactly the case: Our separation is ridiculously good, hence "no convergence"

# Choose the best model and fit to whole dataset ---------------------------------------------------
tab
# classif.ranger.tuned or log_reg

# train tuner_ranger once again
# eigentlich sollte man als Terminator eine bestimmte perfomance oder stagnation in perf wählen
# da aber bei uns fast alle kombinationen 1 ergeben bei auc macht das wohl keinen Sinn
terminator = term("evals", n_evals = 100)
tuner_ranger = AutoTuner$new(
  learner = learner_ranger,
  resampling = resampling_inner_3CV,
  measures = measures, #autotoner nimmt trotzdem nur classif.ce her!?
  tune_ps = param_mtry, 
  terminator = terminator,
  tuner = tuner_grid_search
)
tuner_ranger$train(task_shrooms)

# parameter
tuner_ranger$tuning_instance$archive(unnest = "params")[,c("mtry","auc")]

tuner_ranger$tuning_result
# use those parameters for model
learner_final = lrn("classif.ranger",predict_type = "prob")
learner_final$param_set$values = tuner_ranger$tuning_result$params
learner_final$train(task_shrooms)
pred = learner_final$predict_newdata(newdata = mushrooms_data_test)
pred$score(measures)