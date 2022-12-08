predittori <- read.table(file = "HB_predittori2.csv", header = TRUE, sep = ",")

# Elimino le variabili che non voglio usare
predittori <- predittori[, -1]
predittori <- predittori[, -3]

predittori <- data.frame(scale(predittori, center = TRUE, scale = FALSE))


# FItto il modello sulla variabile Yoon
modAll <- lm(formula = Yoon ~ . - Barton - Corr, data = predittori)
summary(modAll)
modZero <- lm(formula = Yoon ~ 1, data = predittori)
summary(modZero)

# Fitto il modello su Barton
modAll <- lm(formula = Barton ~ . - Yoon - Corr, data = predittori)
summary(modAll)
modZero <- lm(formula = Barton ~ 1, data = predittori)
summary(modZero)

# Scatterplot Matrix

car::scatterplotMatrix(~ Yoon + Barton + Corr + Flame + Social.Use, regLine = list(col = 2),
                       col = 1, smooth = list(col.smooth = 4, col.spread = 4),
                       data = predittori, main = "Scatterplot Matrix")


n <- nrow(predittori)
MASS::stepAIC(modAll, direction = "forward", scope = list(lower = modZero, upper = modAll), k = log(n))
