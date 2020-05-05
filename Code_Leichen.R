# Predictions knn
result_knn = tab_learner_performance$resample_result[[6]]
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


# Ranked Performance------------------------------------------------------------
learner_performance_ranked <- tab_learner_performance[,
                                                      .(learner_id,
                                                        rank_train = rank(-AUC),
                                                        rank_test = rank(MMCE))
                                                      ] %>% 
  arrange(rank_train) # sort in ascending order
learner_performance_ranked

# Tree Plot ------------------------------------------------------------------
# rpart CART implementation:
# rerun the model directly since we cannot access the rpart model in benchmark()
mod_rpart_tree <- rpart::rpart(class ~ ., 
                               data = mushrooms_data)
mod_rpart_tree
# summary(mod_rpart_tree)
mod_rpart_tree$splits
mod_rpart_tree$variable.importance

plot_tree <- rattle::fancyRpartPlot(mod_rpart_tree,
                                    sub = "",
                                    caption = "CART Train Set 1",
                                    palettes = c("Blues",# edible
                                                 "Reds"))# poisonous

rpart::plotcp(mod_rpart_tree) # pruning unnecessary

# Test prediction accuracy

t_pred = predict(mod_rpart_tree, mushrooms_data, type="class")
(confMat <- table(mushrooms_data$class, t_pred))
