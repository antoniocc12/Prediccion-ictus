resumen='''
El accidente cerebrovascular, una enfermedad con morbilidad, discapacidad y mortalidad graves, 
se ha convertido en una de las principales amenazas para la salud pública en todo el mundo.\n
Aunque la patogenia del accidente cerebrovascular aún no está del todo clara, en general se reconoce que 
el accidente cerebrovascular está estrechamente relacionado con indicadores metabólicos anormales tanto para 
el accidente cerebrovascular hemorrágico como para el accidente cerebrovascular isquémico.\nDado que más del 90% de los factores 
de riesgo metabólicos de esta enfermedad pueden controlarse, se debe prestar más atención a la prevención.\nEn el presente trabajo 
se ofrece una herramienta que facilite a profesionales de la salud a predecir posibles ataques de accidente cerebrovascular 
en función de indicadores metabólicos relacionados y proporcionar algún apoyo diagnóstico para la medicina clínica.\n
Para ello, se utilizan varios modelos de aprendizaje automático dirigidos a problemas de clasificación, una optimización 
automática de hiperparámetros de código abierto (Optuna) y su posterior clasificación del individuo en potencial paciente de 
apoplejía o no. Todo ello integrado en una interfaz dinámica de Streamlit en la que, en función del estado de salud del paciente, 
el doctor es capaz de predecir con una alta precisión esta enfermedad.
'''
dataset = '''
En este estudio, el conjunto de datos original de accidente cerebrovascular se recopila de HealthData.gov.\n
Se trata de un conjunto de datos desequilibrado, que contiene 11 características y 
donde se incluyeron 783 casos de accidente cerebrovascular en un total de 43400 muestras registradas, 
lo que representa solo el 1,18% del total.

'''
con1='''
Entre los factores más significativos para sufrir un ataque cerebrovascular encontramos:
- Tener edad avanzada
- El tipo de empleo que se desempeña
- Un alto nivel de glucosa en sangre
- Tener antecedentes clínicos como hipertensión o alguna enfermedad cardíaca
- Ser fumador frecuente
'''
con2='''
Para la predicción médica de apoplejía basada en indicadores fisiológicos y metabólicos, 
se ha propuesto una regresión logística que logra una AUC PR de 8.04%, AUC ROC de 83.07%, exactitud balanceada de 74.82% y 74.52%
de sensibilidad.
'''