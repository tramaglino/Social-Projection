predittori <- read.table(file = "results/Predittori_tutti.csv", header = TRUE, sep = ",")

# Elimino le variabili che non voglio usare
predittori <- predittori[, -1]
predittori <- predittori[, -4]

predittori <- data.frame(scale(predittori, center = TRUE, scale = TRUE))


car::scatterplotMatrix(~ Yoon + Barton + Corr + TFCE + Flame + Social.Use + dx.sx, regLine = list(col = 2),
                       col = 1, smooth = list(col.smooth = 4, col.spread = 4),
                       data = predittori, main = "Scatterplot Matrix")

corrplot::corrplot(cor(predittori), addCoef.col = "grey")


#stepwise regression

n <- nrow(predittori)

modYon <- lm(formula = Yoon ~ (Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog)^2, data = predittori)
modYon_zero <- lm(formula = Yoon ~ 1, data = predittori)
modBarton <- lm(formula = Barton ~ (Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog)^2, data = predittori)
modBarton_zero <- lm(formula = Barton ~ 1, data = predittori)
modCorr <- lm(formula = Corr ~ (Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog)^2, data = predittori)
modCorrzero <- lm(formula = Corr ~ 1, data = predittori)
modTFCE <- lm(formula = TFCE ~ (Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog)^2, data = predittori)
modTFCEzero <- lm(formula = TFCE ~ 1, data = predittori)


stepwise_regYon <- MASS::stepAIC(modYon, direction = "backward", scope = list(lower = modYon_zero, upper = modYon), k = log(n))
stepwise_regBarton <- MASS::stepAIC(modBarton, direction = "backward", scope = list(lower = modBarton_zero, upper = modBarton), k = log(n))
stepwise_regCorr <- MASS::stepAIC(modCorr, direction = "backward", scope = list(lower = modCorrzero, upper = modCorr), k = log(n))
stepwise_regTFCE <- MASS::stepAIC(modTFCE, direction = "backward", scope = list(lower = modTFCEzero, upper = modTFCE), k = log(n))

summary(stepwise_regYon)
summary(stepwise_regBarton)
summary(stepwise_regCorr)
summary(stepwise_regTFCE)
#linearity assumption

par(mfrow = c(2, 2)) # We have 4 predictors
plot(stepwise_regYon, 1)
termplot(stepwise_regYon, partial.resid = TRUE)

#normality
plot(stepwise_regYon, 2)
shapiro.test(modYon$residuals) # p-value = 0.000000000000000222

#yeo-johnson transformation - non-normality patch

YJ <- car::powerTransform(lm(predittori$Yoon ~ 1), family = "yjPower")
(lambdaYJ <- YJ$lambda)
YonTransf <- car::yjPower(U = predittori$Yoon, lambda = lambdaYJ)
predittori <- cbind(predittori, YonTransf)

# Comparison
par(mfrow = c(1, 2))
hist(predittori$Yoon, freq = FALSE, breaks = 10, ylim = c(0, 0.3))
hist(predittori$YonTransf, freq = FALSE, breaks = 10, ylim = c(0, 0.3))
modYon <- lm(formula = YonTransf ~ Flame + Cons.prog + Flame:Cons.prog, data = predittori)
summary(modYon)

shapiro.test(modYon$residuals) # p-value = 0.000000000000000222
plot(modYon, 2)
nortest::lillie.test(modYon$residuals)
qqnorm(modYon$residuals)
qqline(modYon$residuals)


#homoscedasticity
car::ncvTest(modYon)
plot(modYon, 3)
#homoscedasticity ok

#independence
par(mfrow = c(2, 2))
plot(modYon$residuals, type = "o", pch = 19)
lag.plot(modYon$residuals, lags = 1, do.lines = FALSE)

cor(modYon$residuals[-1], modYon$residuals[-length(modYon$residuals)])
car::durbinWatsonTest(modYon)
#independence ok

#multicollinearity

car::vif(modYon, type = 'predictor')

#outliers

plot(modYon, 5)

# multicollinearity ok

#high-leverage points

head(influence(model = modYon, do.coef = FALSE)$hat)

# Another option
h <- hat(x = predittori[, 5:14])

# 1% most influential points
n <- length(predittori[, 5:14])
p <- 1
hist(h, breaks = 20)
abline(v = (qchisq(0.99, df = p) + 1) / n, col = 3)



# PCA regression
#eliminate YonTransformed
predittori <- predittori[, -15]


# Simple call to pcr
library(pls)
modPcr <- pcr(Yoon ~ . - Barton - Corr - TFCE, data = predittori, scale = TRUE)

summary(modPcr)
names(modPcr)
modPcr$coefficients[, , 10]


modPcrCV1 <- pcr(Yoon ~ . -Barton - Corr - TFCE, data = predittori, scale = TRUE,
                 validation = "LOO")

summary(modPcrCV1)
validationplot(modPcrCV1, val.type = "MSEP") # l = 8 gives the minimum CV

modPcrCV10 <- pcr(Yoon ~ . -Barton - Corr - TFCE, data = predittori, scale = TRUE,
                  validation = "CV")
summary(modPcrCV10)
validationplot(modPcrCV10, val.type = "MSEP")

pca_socialpredict <- princomp(x = predittori[, 5:14], cor = TRUE, fix_sign = TRUE)
summary(pca_socialpredict)
plot(pca_socialpredict, type = "l")
biplot(pca_socialpredict, cex = 0.75)

pca3d::pca3d(pca_socialpredict, show.labels = TRUE, biplot = TRUE)
rgl::rglwidget()

#create new dataframe with the scores of PCA

PCA <- data.frame("Yoon" = predittori$Yoon, pca_socialpredict$scores)
modPCA <- lm(Yoon ~ ., data = PCA)
summary(modPCA) 
car::vif(modPCA)



