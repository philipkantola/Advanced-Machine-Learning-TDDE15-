### Lab1 Graphical Models
### Philip Palapelas Kantola
### phika529

### task 1

library(bnlearn)

data("asia")

restart = 1

# create two BNs with same configuration but different seeds
set.seed(12345)
bayesianNetworkFromHC_1 = hc(x= asia, restart = restart, score = "loglik")
set.seed(1)
bayesianNetworkFromHC_2 = hc(x= asia, restart = restart, score = "loglik")

# the two runs are not equal, hence the algorithm can not guarantee global optimum
all.equal(bayesianNetworkFromHC_1,bayesianNetworkFromHC_2)

plot(bayesianNetworkFromHC_1)
plot(bayesianNetworkFromHC_2)


### task 2

predict = function(testData, model, predictors){
  predictionsDiscrete = c()
  
  for(i in 1:dim(testData)[1]){
    
    observedPredictors = c()
    
    for(j in predictors){
      observedPredictors[j] = if(testData[i,j]== "yes") "yes" else "no" 
    }
    evidence = setEvidence(model, nodes =predictors, observedPredictors)
    predictedProbability =  querygrain(evidence, nodes = c("S"))$S["yes"]
    predictionsDiscrete[i] = if(predictedProbability > 0.5) "yes" else "no"
  }
  return (predictionsDiscrete)
}

fitDagAndReturnPredictions = function(network, trainingData,testData, predictors){
  fittedModel = bn.fit(network,trainingData) 
  
  # convert bn.fit object to gRain object
  fittedgRainModel = as.grain(fittedModel)
  # compile graph
  fittedgRainModelCompiled = compile(fittedgRainModel)
  
  predictions = predict(testData, fittedgRainModelCompiled, predictors)
  return(predictions)
}


trainingData = asia[1:4000,]
testData = asia[4001:5000,]

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("RBGL")
BiocManager::install("Rgraphviz")
BiocManager::install("gRain")


# use big number of restarts to achieve better results
restart = 500
set.seed(12345)

bayesianNetworkBIC = hc(x= trainingData, restart = restart, score = "bic")
bayesianNetworkAIC = hc(x= trainingData, restart = restart, score = "aic")

bayesianNetworkLogLik = hc(x= trainingData, restart = restart, score = "loglik")

#compare to true Asia BN
realDag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]", ordering=c("A","S","T","L", "B","E","X","D"))

plot(bayesianNetworkBIC, main="BIC")
plot(bayesianNetworkAIC, main="AIC")
plot(bayesianNetworkLogLik, main="Loglik")
plot(realDag, main="True model")


library(gRain)

predictors = c("A","T","L","B","E","X","D")
predictions = fitDagAndReturnPredictions(bayesianNetworkBIC, trainingData,testData, predictors)

predictionsTrue = fitDagAndReturnPredictions(realDag, trainingData,testData, predictors)

confusionMatrix = table(predictions, testData$S)
confusionMatrixTrue = table(predictionsTrue, testData$S)


### Task 3

# classify S given observations for the Markov Blanket of S

fittedMBNetwork = bn.fit(bayesianNetworkBIC,trainingData)
fittedMBNetworkCompiled = compile(as.grain(fittedMBNetwork))
predictorsMB = mb(fittedMBNetwork, c("S"))
predictionsMB = predict(testData, fittedMBNetworkCompiled, predictorsMB)
confusionMatrixMB = table(predictionsMB, testData$S)


### Task 4


naiveBayesBayesianNetwork <- model2network("[S][A|S][T|S][L|S][B|S][E|S][X|S][D|S]")
plot(naiveBayesBayesianNetwork)

predictionsNaive = fitDagAndReturnPredictions(bayesianNetworkBIC, trainingData,testData, predictors)
confusionMatrixNaive = table(predictionsNaive, testData$S)

