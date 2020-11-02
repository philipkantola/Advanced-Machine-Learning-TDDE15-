### Lab2 Hidden Markov Models
### Philip Palapelas Kantola
### phika529

# Robot walks in ring of 10 unobservable states
# The robot decides with equal probability to stay or move to the next sector.
# Observable: If the robot is in the sector i, then the robot is in the 
# sectors [i − 2, i + 2] with equal probability.

### task 1

# Build a hidden Markov model (HMM) for the scenario described above.

library(HMM)

states = c(1,2,3,4,5,6,7,8,9,10)
symbols = c(1,2,3,4,5,6,7,8,9,10)
startProbs = c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
transitionMatrix = matrix(
  c(0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5,
    0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5
  ),
  nrow = 10, ncol = 10
)
emissionMatrix = matrix(c(0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2,
                         0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0.2,
                         0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0,
                         0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0,
                         0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0,
                         0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 0,
                         0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0,
                         0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2,
                         0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2,
                         0.2, 0.2, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2), nrow=10,ncol=10)
# model
hmmModel = initHMM(states, symbols, startProbs,transitionMatrix,emissionMatrix)

### task 2

# Simulate the HMM for 100 time steps.

hmmSimulation = simHMM(hmmModel, 100)

### task 3

# Discard the hidden states from the sample obtained above. 
# Use the remaining observations to compute the filtered and smoothed probability 
# distributions for each of the 100 time points. Compute also the most probable path.

observations = hmmSimulation$observation

# Take exponent since forward/backward functions return probs in log scale
alpha = exp(forward(hmmModel, observations))
beta = exp(backward(hmmModel, observations))

filteringDistribution = matrix(ncol=dim(alpha)[1],nrow=dim(alpha)[2])
for (t in 1:dim(alpha)[2]){
  filteringDistribution[t,] = alpha[,t]/sum(alpha[,t])
}

smoothingDistribution = matrix(ncol=dim(alpha)[1],nrow=dim(alpha)[2])
for (t in 1:dim(alpha)[2]){
  smoothingDistribution[t,] = (alpha[,t]*beta[,t])/sum(alpha[,t]*beta[,t])
}

# The smoothing distribution obtained from FB-algorithm gives us the probability of any state
# at any time t, however it does not give us the most probable path.
# To obtain the most probable path we must use the Viterbi algorithm.

probablePath = viterbi(hmmModel, observations)
# Most probable path: 
probablePath

### task 4

#Compute the accuracy of the filtered and smoothed probability distributions, and of the
#most probable path. That is, compute the percentage of the true hidden states that are
#guessed by each method

hiddenValues = hmmSimulation$states

# obtain the most probable value for each timestep
filteredDistResult = apply(filteringDistribution,1, which.max)
smoothedDistResult = apply(smoothingDistribution,1, which.max)

table(hiddenValues==probablePath, dnn=c("Viterbi vs hidden values"))
table(hiddenValues==filteredDistResult, dnn=c("Filtered dist vs hidden values"))
table(hiddenValues==smoothedDistResult, dnn=c("Smoothed dist vs hidden values"))
  
### task 5
misClassificationRatesSmoothed = c()
misClassificationRatesFiltered = c()
misClassificationRatesViterbi = c()

  
for(i in 1:10){
  observations = hmmSimulation$observation
  
  # Take exponent since forward/backward functions return probs in log scale
  alpha = exp(forward(hmmModel, observations))
  beta = exp(backward(hmmModel, observations))
  
  filteringDistribution = matrix(ncol=dim(alpha)[1],nrow=dim(alpha)[2])
  for (t in 1:dim(alpha)[2]){
    filteringDistribution[t,] = alpha[,t]/sum(alpha[,t])
  }
  
  smoothingDistribution = matrix(ncol=dim(alpha)[1],nrow=dim(alpha)[2])
  for (t in 1:dim(alpha)[2]){
    smoothingDistribution[t,] = (alpha[,t]*beta[,t])/sum(alpha[,t]*beta[,t])
  }
  probablePath = viterbi(hmmModel, observations)
  filteredDistResult = apply(filteringDistribution,1, which.max)
  smoothedDistResult = apply(smoothingDistribution,1, which.max)

  
  misClassificationRatesSmoothed = append(misClassificationRatesSmoothed,1 - sum(hiddenValues==smoothedDistResult) / length(hiddenValues==smoothedDistResult)) 
  misClassificationRatesFiltered = append(misClassificationRatesFiltered,1 - sum(hiddenValues==filteredDistResult) / length(hiddenValues==filteredDistResult)) 
  misClassificationRatesViterbi = append(misClassificationRatesViterbi,1 - sum(hiddenValues==probablePath) / length(hiddenValues==probablePath)) 
}
# 0.22
smoothedMean = mean(misClassificationRatesSmoothed)
# 0.3
filteredMean = mean(misClassificationRatesFiltered)
# 0.45
viterbiMean = mean(misClassificationRatesViterbi)


### task 6

# Is it true that the more observations you have the better you know where the robot is ?

library(entropy)

# The entropy of a random variable is the average level of "uncertainty" 
# inherent in the variable's possible outcomes.

entropyFiltering = c()
entropySmoothing = c()

for (t in 1:100) {
  entropyFiltering[t] = entropy.empirical(filteringDistribution[t,])
  entropySmoothing[t] = entropy.empirical(smoothingDistribution[t,])
}
plot(entropyFiltering, type="l", xlab="Timestamp",ylab="entropy", main="Entropy of filtering distribution")
print("Mean of entropy of filtering distribution:")
mean(entropyFiltering)
print("Mean of entropy of smoothing distribution:")
mean(entropySmoothing)


### task 7

# Consider any of the samples above of length 100. Compute the probabilities of the
# hidden states for the time step 101.

probabilityVectorT101 = transitionMatrix %*% filteringDistribution[100, ]
probabilityVectorT101
