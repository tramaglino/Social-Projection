predittori <- read.table(file = "results/Predittori_tutti.csv", header = TRUE, sep = ",")

# Elimino le variabili che non voglio usare
predittori <- predittori[, -1]
predittori <- predittori[, -3]

predittori <- data.frame(scale(predittori, center = TRUE, scale = TRUE))


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

# Fitto il modello su Corr
modAll <- lm(formula = Corr ~ . - Yoon - Barton, data = predittori)
summary(modAll)
modZero <- lm(formula = Corr ~ 1, data = predittori)
summary(modZero)
# Scatterplot Matrix

car::scatterplotMatrix(~ Yoon + Barton + Corr + Flame + Social.Use + dx.sx, regLine = list(col = 2),
                       col = 1, smooth = list(col.smooth = 4, col.spread = 4),
                       data = predittori, main = "Scatterplot Matrix")


n <- nrow(predittori)
MASS::stepAIC(modAll, direction = "forward", scope = list(lower = modZero, upper = modAll), k = log(n))


# Path: stepAIC.r

# Aggiungo un modello con interazioni

modYon <- lm(formula = Yoon ~ (Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog)^3, data = predittori)
modYon_zero <- lm(formula = Yoon ~ 1, data = predittori)
modBarton <- lm(formula = Barton ~ (Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog)^3, data = predittori)
modBarton_zero <- lm(formula = Barton ~ 1, data = predittori)
modCorr <- lm(formula = Corr ~ (Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog)^3, data = predittori)
modCorrzero <- lm(formula = Corr ~ 1, data = predittori)

n <- nrow(predittori)
MASS::stepAIC(modYon, direction = "backward", scope = list(lower = modYon_zero, upper = modYon), k = log(n))
MASS::stepAIC(modBarton, direction = "backward", scope = list(lower = modBarton_zero, upper = modBarton), k = log(n))
MASS::stepAIC(modCorr, direction = "backward", scope = list(lower = modCorrzero, upper = modCorr), k = log(n))


summary(lm(formula = Corr ~ Flame + Social.Use + Em.Stability + dx.sx + 
    Agreeableness + Conscientiousness + Cons.prog + Flame:Social.Use + 
    Flame:Agreeableness + Flame:Conscientiousness + Flame:Cons.prog + 
    Social.Use:Em.Stability + Social.Use:dx.sx + Social.Use:Conscientiousness + 
    Social.Use:Cons.prog + Em.Stability:dx.sx + dx.sx:Conscientiousness + 
    Conscientiousness:Cons.prog + Flame:Social.Use:Conscientiousness, 
    data = predittori))

summary(lm(Barton ~ Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + 
    Conscientiousness + Cons.prog + Flame:Social.Use + Flame:Em.Stability + 
    Flame:dx.sx + Flame:Conscientiousness + Flame:Cons.prog + 
    Social.Use:Em.Stability + Social.Use:dx.sx + Social.Use:Conscientiousness + 
    Social.Use:Cons.prog + Em.Stability:dx.sx + Em.Stability:Agreeableness + 
    dx.sx:Agreeableness + dx.sx:Cons.prog + Agreeableness:Conscientiousness + 
    Conscientiousness:Cons.prog + Flame:Social.Use:Conscientiousness + 
    Social.Use:dx.sx:Cons.prog, data = predittori))
summary(lm(Yoon ~ Flame + Social.Use + Em.Stability + Agreeableness + Conscientiousness + 
    Cons.prog + Flame:Social.Use + Flame:Agreeableness + Flame:Conscientiousness + 
    Flame:Cons.prog + Social.Use:Em.Stability + Social.Use:Conscientiousness + 
    Conscientiousness:Cons.prog + Flame:Social.Use:Conscientiousness, data= predittori))

summary(modYon)
summary(modBarton)
summary(modCorr)

plot(Yoon ~ Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog, data = predittori)
abline(modYon, col = "red") 
abline(modYon_zero, col = "blue")

plot(Barton ~ Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog, data = predittori)
abline(modBarton, col = "red")
abline(modBarton_zero, col = "blue")

plot(Corr ~ Flame + Social.Use + Em.Stability + dx.sx + Agreeableness + Conscientiousness + Cons.prog, data = predittori)
abline(modCorr, col = "red")
abline(modCorrzero, col = "blue")
