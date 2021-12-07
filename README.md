# Pràctica Kaggle APC UAB 2021-22

### Nom: Martí
### DATASET: Student Alcohol Consumption
### URL: [kaggle](https://www.kaggle.com/uciml/student-alcohol-consumption)


## Resum
Aquest repositori conté un seguit de mètodes que es poden utilitzar per resoldre problemes d'aprenentatge computacional. Tant el codi com els resultats estan en un Jupyter Notebook per tal de poder mostrar-se amb claredat. 

La base de dades amb la que es treballadrà és l'anomenada [Student Alcohol Consumption](https://www.kaggle.com/uciml/student-alcohol-consumption), extreta del portal kaggle.

Els principals punts en els que està estructurat el treball són:
1. Explicació dels atributs més importants i atribut a predir.
2. Aplcació dels mètodes d'aprenentatge computacional.
3. Presentació de resultats i conclusions.

## Objectius del dataset
Per una banda es vol veure si, tal i com diu el títol, hi ha relació entre el consum d'alcohol i les notes. Per altra banda es vol predir la nota final d'un alumne basant-se en la resta d'atributs.

## Exeperiments
Durant aquesta pràctica s'ha fet un seguit de proves i experiments.
Primer s'han implementat els següents models sense parar atenció als hiperparàmtres, simplement per veure de primeres quin donava més bon resultat.
1. Decision Tree Regressor:
2. Ridge
3. Linear Regression:
4. Lasso

També s'ha utilitzat el "Principal Component Analysis" per veure fins a quina dimensió es podien reduir les mosters sense perdre capacitat predictiva.

Una altra proba que s'ha portat a terme és l'anomenada "Hyperparameter Search". S'ha utilitzat el mètode "Bayesian Optimization" i els resultats han sigut satisfactoris.

Per intentar millorar més les prediccions, també s'ha implementat mètodes de Boosting. Igual que amb la proba superior, les prediccions han millorat.

Per últim, s'ha intentat ajuntar Boosting amb Hyperparameter Search, però els resultats han sigut pràcticament els mateixos.
