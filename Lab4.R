### Task 1
posteriorGP <- function(x, y, XStar, sigmaNoise, sigmaF, l, k) {
  K = k(x, x, sigmaF, l)
  Kstar = k(x, XStar, sigmaF, l)
  Kstarstar= k(XStar, XStar, sigmaF, l)
  
  oneDiag = diag(1, length(x))
  L = t(chol(K + (sigmaNoise^2)*oneDiag))
  alpha = solve(a = t(L),b = solve(a = L, b = t(y)))
  
  fStar = t(Kstar) %*% alpha
  v = solve(L, Kstar)
  bigV = Kstarstar - t(v) %*% v
  n = length(y)
  logSum = 0
  for(i in (1:n)) {
    logSum = logSum + log(L[i,i]) - n/2*log(2*pi)
  }
  loglikelihood = (-1/2)*t(y)*alpha - logSum
  return(list('mean' = fStar, 'variance' = diag(bigV), 'log_marginal_likelihood' = loglikelihood))
}

SquaredExpKernel <- function(x1,x2,sigmaF=1,l=3){
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA,n1,n2)
  for (i in 1:n2){
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/l)^2)
  }
  return(K)
}

plotPostResults <- function(x, y, xTest, prediction, variance) {
  upperBand = prediction + 1.96*sqrt(variance)
  lowerBand = prediction - 1.96*sqrt(variance)
  
  
  plot(x, y, type="p", 
       ylim=c(min(min(y), min(lowerBand)),
              max(max(y), max(upperBand))))
  lines(xTest,prediction, col = "red", lwd = 3)
  lines(xTest, upperBand, col = "blue", lwd = 2)
  lines(xTest, lowerBand, col = "blue", lwd = 2)
  points(x, y, col='green')
}


plotPostResults2 <- function(x, y, xTest, prediction1, prediction2, variance1, variance2) {
  upperBand1 = prediction1 + 1.96*sqrt(variance1)
  lowerBand1 = prediction1 - 1.96*sqrt(variance1)
  
  upperBand2 = prediction2 + 1.96*sqrt(variance2)
  lowerBand2 = prediction2 - 1.96*sqrt(variance2)
  
  
  
  plot(x, y, type="p", 
       ylim=c(min(min(y), min(lowerBand1), min(lowerBand2)),
              max(max(y), max(upperBand1), max(upperBand2))))
  lines(xTest,prediction1, col = "red", lwd = 3)
  lines(xTest, upperBand1, col = "blue", lwd = 2)
  lines(xTest, lowerBand1, col = "blue", lwd = 2)
  lines(xTest,prediction2, col = "orange", lwd = 3)
  lines(xTest, upperBand2, col = "brown", lwd = 2)
  lines(xTest, lowerBand2, col = "brown", lwd = 2)
  points(x, y, col='green')
}

### 1.2
# Plotting one draw
sigmaF <- 1
l <- 0.3
nSim <- 1

fSim <- c(0.719)
x = c(0.4)
y = t(matrix(c(0.719)))
xTest = seq(-1,1,length=10)
sigmaNoise = 0.1

post_result = posteriorGP(x, y, xTest, sigmaNoise, sigmaF, l, SquaredExpKernel)

plotPostResults(x, y, xTest, post_result, sigmaF, l)

### 1.3
sigmaF <- 1
l <- 0.3
nSim <- 1

x = c(0.4, -0.6)
y = t(matrix(c(0.719, -0.044)))
xTest = seq(-1,1,length=10)
sigmaNoise = 0.1

post_result = posteriorGP(x, y, xTest, sigmaNoise, sigmaF, l, SquaredExpKernel)

plotPostResults(x, y, xTest, post_result, sigmaF, l)

### 1.4
sigmaF <- 1
l <- 0.3
nSim <- 1

x = c(-1.0, -0.6, -0.2, 0.4, 0.8)
y = t(matrix(c(0.768, -0.044, -0.940, 0.719, -0.664)))
xTest = seq(-1,1,length=10)
sigmaNoise = 0.1

post_result = posteriorGP(x, y, xTest, sigmaNoise, sigmaF, l, SquaredExpKernel)

plotPostResults(x, y, xTest, post_result, sigmaF, l)
### 1.5
sigmaF <- 1
l <- 1
nSim <- 1

x = c(-1.0, -0.6, -0.2, 0.4, 0.8)
y = t(matrix(c(0.768, -0.044, -0.940, 0.719, -0.664)))
xTest = seq(-1,1,length=10)
sigmaNoise = 0.1

post_result = posteriorGP(x, y, xTest, sigmaNoise, sigmaF, l, SquaredExpKernel)

plotPostResults(x, y, xTest, post_result, sigmaF, l)

### Task 2


### 2.1
squareKernel <- function(sigmaf = 1, ell = 1) 
{
  rval <- SquaredExpKernel <- function(x, y = NULL) {
    n1 <- length(x)
    n2 <- length(y)
    K <- matrix(NA,n1,n2)
    for (i in 1:n2){
      K[,i] <- sigmaf^2*exp(-0.5*( (x-y[i])/ell)^2)
    }
    return(K)
  }
  class(rval) <- "kernel"
  return(rval)
} 
library(lubridate)
library(kernlab)
library(AtmRay)

temp_data = read.csv(file="c:\\ml_school\\TDDE15\\TDDE15-Advanced-Machine-Learning\\lab 4\\TempTullinge.csv", header=TRUE, sep=";")
dates2 <- as.Date(temp_data$date,format="%d/%m/%Y")
dates = temp_data$date
temps = temp_data$temp
time = 1:length(dates)
days <- yday(dates2)

temps_5 = temps[seq(1, length(temps), 5)]
days_5 = days[seq(1, length(days), 5)]
time_5 = time[seq(1, length(time), 5)]

days <- yday(dates2)

x1 = c(1)
x2 = c(2)

X = t(c(1, 3, 4))
Xstar = t(c(2, 3, 4))

squareFunc = squareKernel(sigmaf = 1, ell = 1)
squareFunc(x1, x2)
K = kernelMatrix(kernel = MaternFunc, x = X, y = Xstar)
K

### 2.2

regFit = lm(temps_5 ~ time_5 + time_5^2)
res_std = sd(regFit$residual)

sigmaf = 20
ell = 0.2
GPfit <- gausspr(x = time_5, y = temps_5, 
                 kernel = squareKernel,
                 kpar = list(sigmaf=sigmaf, ell=ell),
                 var = res_std^2)
meanPred1 <- predict(GPfit, time)
plot(time_5, temps_5)
lines(time_5, meanPred1, col="purple", lwd = 2)

### 2.3
post_result1 <- posteriorGP(x = scale(time_5),
                            y = t(temps_5),
                            XStar = scale(seq(from = 1, to = 365*6, by = 1)),
                            sigmaNoise = res_std,
                            sigmaF = sigmaf,
                            l=ell,
                            k=SquaredExpKernel)
plotPostResults(time, t(temps), seq(from = 1, to = 365*6, by = 1), meanPred1, post_result1$variance)

### 2.4


regFit = lm(temps_5 ~ days_5 + days_5^2)
res_std = sd(regFit$residual)

sigmaf = 20
ell = 0.2

GPfit <- gausspr(x = days_5, y = temps_5, 
                 kernel = squareKernel,
                 kpar = list(sigmaf=sigmaf, ell=ell),
                 var = res_std^2)
meanPred2 <- predict(GPfit, days)
plot(days_5, temps_5)
lines(days, meanPred, col="purple", lwd = 2)
post_result2 <- posteriorGP(x = days_5,
                            y = t(temps_5),
                            XStar = days,
                            sigmaNoise = res_std,
                            sigmaF = sigmaf,
                            l=ell,
                            k=SquaredExpKernel)
plotPostResults2(time, t(temps), time, meanPred1, meanPred2, post_result1$variance, post_result2$variance)

### 2.5

periodic 

periodicKernel <- function(sigmaf, l1, l2, d) {
  rval <- periodic <- function(x1, x2 = NULL) {
    return(sigmaf^2*exp(-2*sin(pi*abs(x1 - x2)/d)^2/(l1^2))
           *exp(-1/2*abs(x1 - x2)^2/(l2^2)))
  }
  class(rval) <- "kernel"
  return(rval)
}

regFit = lm(temps_5 ~ time_5 + time_5^2)
res_std = sd(regFit$residual)

sigmaf = 20
l1 = 1
l2 = 10
d = 365/sd(time)

GPfit <- gausspr(x = time_5, y = temps_5, 
                 kernel = periodicKernel,
                 kpar = list(sigmaf=sigmaf, l1=l1, l2=l2, d=d),
                 var = res_std^2)
meanPred3 <- predict(GPfit, time_5)

post_result3 <- posteriorGP(x = time_5,
                            y = t(temps_5),
                            XStar = seq(from = 1, to = 365*6, by = 5),
                            sigmaNoise = res_std,
                            sigmaF = sigmaf,
                            l=ell,
                            k=SquaredExpKernel)
plotPostResults(time_5, t(temps_5), seq(from = 1, to = 365*6, by = 5), meanPred3, post_result3$variance)

### Task 3

data <- read.csv("c:\\ml_school\\TDDE15\\TDDE15-Advanced-Machine-Learning\\lab 4\\banknoteFraud.csv", header=FALSE, sep=",") 
names(data) <- c("varWave","skewWave","kurtWave","entropyWave","fraud") 
data[,5] <- as.factor(data[,5])

# Sampling data given seed 111
set.seed(111); 
SelectTraining <- sample(1:dim(data)[1], size = 1000, replace = FALSE)

#dividing data into train and test data
selectedData = data[SelectTraining[1:500],]
testData = data[SelectTraining[501:1000],]

#Fitting the gaussian process with the training data
GPfit <- gausspr(x = selectedData[,1:2], y = selectedData[,5])

#Setting the limits for the plots based on the data
varLim = c(min(data[,1]), max(data[,1]))
skewLim = c(min(data[,2]), max(data[,2]))

#Creating sequences to be able to map the predicted values from the model at different input values
seqVar = seq(from = varLim[1], to = varLim[2], by = (varLim[2] - varLim[1])/20)
seqSkew = seq(from = skewLim[1], to = skewLim[2], by = (skewLim[2] - skewLim[1])/20)

# Creating a NxN grid based on the two sequences oflength N
predSeq = matrix(nrow=length(seqVar)*length(seqSkew), ncol=2)
for(i in 1:length(seqVar)) {
  for(j in 1:length(seqSkew)) {
    predSeq[(i - 1)*length(seqSkew) + j,] = c(seqVar[i], seqSkew[j])
  }
}
pred <- predict(GPfit, predSeq, type="probabilities")

#Dividing the points into not fraud and fraud points
data_df = data.frame(selectedData)
names(data_df) <- c("varWave","skewWave","kurtWave","entropyWave","fraud")
true_fraud = selectedData[which(selectedData$fraud == 1),]
false_fraud = selectedData[which(selectedData$fraud == 0),]

#Creating the plots of the data
contour(x = seqVar, y = seqSkew, z = matrix(pred[,2], length(seqVar), byrow = TRUE))
points(true_fraud[,1], true_fraud[,2], col='blue')
points(false_fraud[,1], false_fraud[,2], col='red')
###

predTest <- predict(GPfit, testData[,1:2])

true_vals = testData[,5]

cov_matrix = table(predTest, true_vals)
accuracy2 = sum(diag(cov_matrix))/length(true_vals)
print(accuracy2)

###

GPfit <- gausspr(x = selectedData[,1:4], y = selectedData[,5])

predTest <- predict(GPfit, testData[,1:4])

true_vals = testData[,5]

cov_matrix = table(predTest, true_vals)
accuracy4 = sum(diag(cov_matrix))/length(true_vals)
print(accuracy4)