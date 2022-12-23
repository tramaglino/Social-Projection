predittori <- read.table(file = "results/Predittori_tutti.csv", header = TRUE, sep = ",")

# Elimino le variabili che non voglio usare
predittori <- predittori[, -1]
predittori <- predittori[, -4]

predittori <- data.frame(scale(predittori, center = TRUE, scale = TRUE))

Barton <- predittori$Barton
Yoon <- predittori$Yoon
Corr <- predittori$Corr
TFCE <- predittori$TFCE


predittori <- subset(predittori, select=-c(Yoon,Barton, Corr, TFCE))
x <- model.matrix(Corr ~ 0 + .^2, data = predittori) 

x <- x[complete.cases(x), , drop = FALSE]

y <- Corr

library(glmnet)
ridgeMod <- glmnet(x = x, y = y, alpha = 0)


plot(ridgeMod, xvar = "norm", label = TRUE)
# xvar = "norm" is the default: L1 norm of the coefficients sum_j abs(beta_j)

# Versus lambda
plot(ridgeMod, label = TRUE, xvar = "lambda")

# Versus the percentage of deviance explained -- this is a generalization of the
# R^2 for generalized linear models. Since we have a linear model, this is the
# same as the R^2
plot(ridgeMod, label = TRUE, xvar = "dev")
# The maximum R^2 is above 0.35

plot(log(ridgeMod$lambda), ridgeMod$dev.ratio, type = "l",
     xlab = "log(lambda)", ylab = "R2")
ridgeMod$dev.ratio[length(ridgeMod$dev.ratio)]

# The maximum R^2 is above 0.35 but with the penalty the fitness decreases

# The coefficients are in the matrix ridgeMod$beta
# Zoom in path solution
plot(ridgeMod, label = TRUE, xvar = "lambda",
     xlim = log(ridgeMod$lambda[15]) + c(-2, 2), ylim = c(-0.03, 0.03))
abline(v = log(ridgeMod$lambda[15]))
points(rep(log(ridgeMod$lambda[15]), nrow(ridgeMod$beta)), ridgeMod$beta[, 15],
       pch = 16, col = 1:6)

# The squared l2-norm of the coefficients decreases as lambda increases
plot(log(ridgeMod$lambda), sqrt(colSums(ridgeMod$beta^2)), type = "l",
     xlab = "log(lambda)", ylab = "l2 norm")



# 10-fold cross-validation. Change the seed for a different result
set.seed(12345)
kcvRidge <- cv.glmnet(x = x, y = y, alpha = 0, nfolds = 10)

# The lambda that minimizes the CV error is
kcvRidge$lambda.min

# The minimum CV error
min(kcvRidge$cvm)

range(kcvRidge$lambda)
lambdaGrid <- 10^seq(log10(kcvRidge$lambda[1]), log10(0.1),
                     length.out = 150) # log-spaced grid
kcvRidge2 <- cv.glmnet(x = x, y = y, nfolds = 10, alpha = 0,
                       lambda = lambdaGrid)

# Much better
plot(kcvRidge2)
kcvRidge2$lambda.min

kcvRidge2$lambda.1se


plot(kcvRidge2)
indMin2 <- which.min(kcvRidge2$cvm)
abline(h = kcvRidge2$cvm[indMin2] + c(0, kcvRidge2$cvsd[indMin2]))

ncvRidge <- cv.glmnet(x = x, y = y, alpha = 0, nfolds = nrow(predittori),
                      lambda = lambdaGrid)

plot(ncvRidge)

modRidgeCV <- kcvRidge2$glmnet.fit


# Inspect the best models
plot(modRidgeCV, label = TRUE, xvar = "lambda")
abline(v = log(c(kcvRidge2$lambda.min, kcvRidge2$lambda.1se)))

predict(modRidgeCV, type = "coefficients", s = kcvRidge2$lambda.min)

plot(log(modRidgeCV$lambda),
     predict(modRidgeCV, type = "response", newx = x[1, , drop = FALSE]),
     type = "l", xlab = "log(lambda)", ylab = " Prediction")
