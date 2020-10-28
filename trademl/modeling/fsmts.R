library(DBI)
library(RMySQL)
library(anytime)
require(fsMTS)
require(plot.matrix)
require(svMisc)
require(MTS)

The goal of the package is to finde alpha and apply profitable strategies to paper trading using Alpaca.

# IMPORT DATA -------------------------------------------------------------

# connect to database
connection <- function() {
  con <- DBI::dbConnect(RMySQL::MySQL(),
                        host = "91.234.46.219",
                        port = 3306L,
                        dbname = "odvjet12_market_data_usa",
                        username = 'odvjet12_mislav',
                        password = 'Theanswer0207',
                        Trusted_Connection = "True")
}


# import ohlcv and python frac diff data
db <- connection()
contract <- dbGetQuery(db, 'SELECT date, open, high, low, close, volume FROM SPY')
invisible(dbDisconnect(db))
contract$date <- anytime::anytime(contract$date)

# convert to xts and generate lowe frequncy series
contract_xts <- xts::xts(contract[, -1], order.by=contract$date, unique=TRUE, tzone='America/Chicago')
contract_ohlcv <- contract_xts[, c('open', 'high', 'low', 'close')]
colnames(contract_ohlcv) <- c('open', 'high', 'low', 'close')
contract_daily <- xts::to.daily(contract_ohlcv)
contract_hourly <- xts::to.hourly(contract_ohlcv)
contract_weekly <- xts::to.weekly(contract_ohlcv)
contract_monthly <- xts::to.monthly(contract_ohlcv)
close_daily <- contract_daily$contract_ohlcv.Close
close_hourly <- contract_hourly$contract_ohlcv.Close
close_weekly <- contract_weekly$contract_ohlcv.Close
close_monthly <- contract_monthly$contract_ohlcv.Close


# FEATURE SELECTION ----------------------------------------------------------------

# onvert to matrix and scale
close <- contract$close

m <- as.matrix(contract[, -1])
m <- base::scale(m)

# parametres for functions
max.lag <- 10
show.progress = TRUE

# calculate shortest matrix
k <- ncol(close)
shortest <- matrix(rexp(k*k, rate = 0.2), nrow=k)
shortest <- shortest-diag(k)*shortest
colnames(shortest) <- colnames(close)
rownames(shortest) <- colnames(close)
plot(shortest, digits=2, col=rev(heat.colors(10)), key=NULL, 
     main="Shortest distances between nodes")

# use all feature selection methods
mIndep <- fsMTS(close, max.lag=max.lag, method="ownlags",show.progress=show.progress)
mCCF <- fsMTS(close, max.lag=max.lag, method="CCF", show.progress=show.progress)
mDistance <- fsMTS(close, max.lag=max.lag, method="distance", shortest = shortest, step = 5, show.progress=show.progress)  # MEMORY INTENSIVE
mGLASSO.global <- fsMTS(close, max.lag=max.lag,method="GLASSO", rho = 0.1,show.progress=show.progress, localized = FALSE)
mLARS <- fsMTS(close, max.lag=max.lag,method="LARS",show.progress=show.progress)
mRF.global <- fsMTS(close, max.lag=max.lag,method="RF",show.progress=show.progress, localized = FALSE)
mMI.global <- fsMTS(m, max.lag=max.lag,method="MI",show.progress=show.progress, localized= FALSE)
mPSC <- fsMTS(m, max.lag=max.lag,method="PSC",show.progress=show.progress)

# merge
mlist <- list(Independent = mIndep,
              Distance = mDistance,
              CCF = mCCF,
              GLASSO.global = mGLASSO.global,
              LARS = mLARS,
              RF.global = mRF.global,
              MI.global = mMI.global,
              PSC=mPSC)

# 
th<-0.1
mE1 <- fsEnsemble(mlist, threshold = th, method="ranking")
plot(mE1, col=rev(heat.colors(10)), key=NULL, 
     main="Ensemble feature selection  using Ranking")

dim(mE1)
mE1[, 1]
data[mE1[, 1], 1]

rownames(mlist$Independent)[mE1[, 1]]



data(traffic)
data <- scale(traffic$data[,-1])
max.lag <- 3
show.progress = F
traffic$shortest


mGLASSO.global <- fsMTS(m, max.lag=max.lag,method="GLASSO", rho = 0.1,show.progress=show.progress, localized = FALSE)
mts = m
max.lag=3
method="GLASSO"
rho = 0.1
show.progress=show.progress
localized = FALSE

k<-ncol(mts)
cors <- NULL
for (l in 1:max.lag) {
  dat_lag <- cbind(mts[1:(nrow(dat)-l),], mts[(l+1):nrow(mts),])
  cors_l <- cor(stats::cor(dat_lag))
  cors_l <- cors_l[, (k+1):ncol(cors_l)]
  colnames(cors_l) <- paste(colnames(cors_l), '_', l)
  cors <- cbind(cors, cors_l)
}

dim(dat[l:nrow(dat),])
head(dat[l:nrow(dat),])
head(dat)
dim(dat[1:(nrow(dat)-l),])
head(dat[1:(nrow(dat)-l),])
dim(dat[-nrow(dat),])
dim(mts)
dim(mts[-c(1:l),])
head(dat)
cors[[2]]

fsGLASSO <- function(mts, max.lag, rho, absolute = TRUE, show.progress = TRUE, localized = FALSE) {
  k<-ncol(mts)
  if (localized){
    res<-matrix(0, k*max.lag, k)
    for (i in 1:k){
      dat <- composeYX(mts, i, max.lag)
      dat.cov<-stats::cor(dat)
      gl<-glasso::glasso(dat.cov, rho=rho, penalize.diagonal=FALSE)
      links<-gl$wi[1,-1]
      res[,i] <- links
      if (show.progress) svMisc::progress(100*i/k)
    }
    res <- fsNames(res, mts, max.lag)
  }else{
    dat<-mts
    for (l in 1:max.lag){
      dat<-cbind(dat[-nrow(dat),], mts[-c(1:l),])
    }
    dat.cov<-stats::cor(dat)
    gl<-glasso::glasso(dat.cov, rho=rho, penalize.diagonal=FALSE)
    res<-gl$wi[-c(1:k),1:k]
    res <- fsNames(res, mts, max.lag)
  }
  if (absolute) res <- abs(res)
  
  return (res)
}


# BIGVAR ------------------------------------------------------------------


# N-fold cross validation for VAR
# Y: data
# nfolds: number of cross validation folds
# struct: penalty structure
# p: lag order 
# nlambdas: number of lambdas:
# gran1: depth of lambda grid
# seed: set to make it reproducible
NFoldcv <- function(Y,nfolds,struct,p,nlambdas,gran1,seed)
{
  Y <- m
  MSFE <- matrix(0,nrow=nrow(Y),ncol=nfolds)
  A <- constructModel(Y,p,struct=struct,gran=c(gran1,nlambdas),verbose=F)
  # construct lag matrix                                      
  Z1 <- VARXLagCons(Y,X=NULL,p=p)
  
  trainZ <- Z1$Z[2:nrow(Z1$Z),]
  
  trainY <- matrix(Y[(p+1):nrow(Y),],ncol=ncol(Y)) 
  set.seed(seed)
  inds <- sample(nrow(trainY))
  
  trainY <- trainY[inds,]
  
  trainZ <- trainZ[,inds]
  # fit on all training data to get penalty grid
  B <- BigVAR.est(A)
  lambda.grid <- B$lambda
  folds <- cut(seq(1,nrow(trainY)),breaks=nfolds,labels=FALSE)
  
  MSFE <- matrix(0,nrow=nfolds,ncol=nlambdas)
  for(i in 1:nfolds){
    
    test <- trainY[which(folds==i),]
    train <- trainY[which(folds!=i),]
    testZ <-t(t(trainZ)[which(folds!=i),])
    
    B=BigVAR.fit(train,p=p,lambda=lambda.grid,struct=struct)
    
    #iterate over lambdas
    for(j in 1:nlambdas){
      
      MSFETemp <- c()
      for(k in 1:nrow(test))    {
        tempZ <- testZ[,k,drop=FALSE]
        bhat <- matrix(B[,2:dim(B)[2],j],nrow=ncol(Y),ncol=(p*ncol(Y)))
        preds <- B[,1,j]+bhat%*%tempZ
        
        MSFETemp <- c(MSFETemp,sum(abs(test[k,]-preds))^2)
        
      }
      MSFE[i,j] <- mean(MSFETemp)
      
      
    }
    
  }
  
  return(list(MSFE=MSFE,lambdas=lambda.grid))
}

library(BigVAR)

# 10 fold cv
MSFEs<-NFoldcv(m, nfolds=5, "Basic", p=5, nlambdas=10, gran1=50, seed=123)
# choose smaller lambda in case of ties (prevents extremely sparse solutions)
opt=MSFEs$lambda[max(which(colMeans(MSFEs$MSFE)==min(colMeans(MSFEs$MSFE))))]
opt



mGLASSO.global <- fsMTS(m, max.lag=max.lag,method="GLASSO", rho = 0.1,show.progress=show.progress, localized = FALSE)
mts = m
max.lag=max.lag
method="GLASSO"
rho = 0.1
show.progress=show.progress
localized = FALSE


k <- ncol(mts)
dat<-mts
for (l in 1:max.lag){
  dat <- cbind(dat[-nrow(dat),], mts[-c(1:l),])
}


fsGLASSO <- function(mts, max.lag, rho, absolute = TRUE, show.progress = TRUE, localized = FALSE) {
  k<-ncol(mts)
  if (localized){
    res<-matrix(0, k*max.lag, k)
    for (i in 1:k){
      dat <- composeYX(mts, i, max.lag)
      dat.cov<-stats::cor(dat)
      gl<-glasso::glasso(dat.cov, rho=rho, penalize.diagonal=FALSE)
      links<-gl$wi[1,-1]
      res[,i] <- links
      if (show.progress) svMisc::progress(100*i/k)
    }
    res <- fsNames(res, mts, max.lag)
  }else{
    dat<-mts
    for (l in 1:max.lag){
      dat<-cbind(dat[-nrow(dat),], mts[-c(1:l),])
    }
    dat.cov<-stats::cor(dat)
    gl<-glasso::glasso(dat.cov, rho=rho, penalize.diagonal=FALSE)
    res<-gl$wi[-c(1:k),1:k]
    res <- fsNames(res, mts, max.lag)
  }
  if (absolute) res <- abs(res)
  
  return (res)
}


# BIGVAR ------------------------------------------------------------------


# N-fold cross validation for VAR
# Y: data
# nfolds: number of cross validation folds
# struct: penalty structure
# p: lag order 
# nlambdas: number of lambdas:
# gran1: depth of lambda grid
# seed: set to make it reproducible
NFoldcv <- function(Y,nfolds,struct,p,nlambdas,gran1,seed)
{
  Y <- m
  MSFE <- matrix(0,nrow=nrow(Y),ncol=nfolds)
  A <- constructModel(Y,p,struct=struct,gran=c(gran1,nlambdas),verbose=F)
  # construct lag matrix                                      
  Z1 <- VARXLagCons(Y,X=NULL,p=p)
  
  trainZ <- Z1$Z[2:nrow(Z1$Z),]
  
  trainY <- matrix(Y[(p+1):nrow(Y),],ncol=ncol(Y)) 
  set.seed(seed)
  inds <- sample(nrow(trainY))
  
  trainY <- trainY[inds,]
  
  trainZ <- trainZ[,inds]
  # fit on all training data to get penalty grid
  B <- BigVAR.est(A)
  lambda.grid <- B$lambda
  folds <- cut(seq(1,nrow(trainY)),breaks=nfolds,labels=FALSE)
  
  MSFE <- matrix(0,nrow=nfolds,ncol=nlambdas)
  for(i in 1:nfolds){
    
    test <- trainY[which(folds==i),]
    train <- trainY[which(folds!=i),]
    testZ <-t(t(trainZ)[which(folds!=i),])
    
    B=BigVAR.fit(train,p=p,lambda=lambda.grid,struct=struct)
    
    #iterate over lambdas
    for(j in 1:nlambdas){
      
      MSFETemp <- c()
      for(k in 1:nrow(test))    {
        tempZ <- testZ[,k,drop=FALSE]
        bhat <- matrix(B[,2:dim(B)[2],j],nrow=ncol(Y),ncol=(p*ncol(Y)))
        preds <- B[,1,j]+bhat%*%tempZ
        
        MSFETemp <- c(MSFETemp,sum(abs(test[k,]-preds))^2)
        
      }
      MSFE[i,j] <- mean(MSFETemp)
      
      
    }
    
  }
  
  return(list(MSFE=MSFE,lambdas=lambda.grid))
}

library(BigVAR)

# 10 fold cv
MSFEs<-NFoldcv(m, nfolds=5, "Basic", p=5, nlambdas=10, gran1=50, seed=123)
# choose smaller lambda in case of ties (prevents extremely sparse solutions)
opt=MSFEs$lambda[max(which(colMeans(MSFEs$MSFE)==min(colMeans(MSFEs$MSFE))))]
opt





# dseg --------------------------------------------------------------------



  
