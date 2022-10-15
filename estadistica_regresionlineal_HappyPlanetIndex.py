
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import scipy.stats

df = pd.read_csv ("copia_happyplace.csv")

# Guardar las listas como numpy arrays
x = np.array (df ["GDPperCapita"])
y = np.array (df ["Happiness"])

# Creamos un objeto de la clase LinerarRegresion
regresion_lineal = LinearRegression ()

# Ajustamos los datos a una línea recta
regresion_lineal.fit (x.reshape (-1,1) , y)

# Obtenemos la pendiente
b = regresion_lineal.coef_

# Obtenemos la intersección
a = regresion_lineal.intercept_

print ("Ecuacion es y = bx + a ")
print ("b = " , regresion_lineal.coef_ [0])
print ("a = " , regresion_lineal.intercept_)


# plt.scatter (x , y)


fig , ax = plt.subplots()


# Etiquetas

ax.set_xlabel ("GDPperCapita" , fontsize = 12)
ax.set_ylabel ("Happiness" , fontsize = 12)
ax.set_title ("Regresión Lineal" , fontsize = 15)

# Dar color a las líneas de abajo, arriba, a la etiqueta de x, y su numeración
ax.spines ['bottom'].set_color ('red')
ax.spines ['top'].set_color ('red')
ax.xaxis.label.set_color ('red')
ax.tick_params (axis='x', colors='red')

# Da color a las líneas de la derecha, izquierda, a la etiqueta de y,
# así como su numeración
ax.spines ['right'].set_color ('orange')
ax.spines ['left'].set_color ('orange')
ax.yaxis.label.set_color ('blue')
ax.tick_params (axis='y', colors='orange')

ax = sns.regplot (x, y, x_ci = 98)
plt.xlim (0,60228)
plt.ylim (0,8.5)
plt.show ()


# _____________________________ Validación del Modelo _________________________

# 1.- Prueba de hipótesis
y_pred=np.array(b*x+a)
x_prom=sum(x)/len(x)
s=math.sqrt(sum((y-y_pred)**2/(len(y)-2)))
den=math.sqrt(sum((x-x_prom)**2))
residuales=np.array(y-y_pred)

print("El valor de s", s)
print("Denominador", den)
t=b/(s/den)
grados_de_libertad=len(y)-2
print(t)

p_value=2*scipy.stats.t.sf(abs(t),grados_de_libertad)
print("El p-value es", p_value)

# 2.- Residuales - Patrón
plt.scatter(x,residuales)

# 3.- Promedio de los residuales
prom_residuales=np.sum(residuales)/len(residuales)

print("Suma residuales", np.sum(residuales))
print("Len residuales", len(residuales))
print("Promedio de los residuales", prom_residuales)


# _____________________________ Histograma ____________________________________
fig2,ax2=plt.subplots()
N=math.ceil(math.log(len(x))/math.log(2))
plt.hist(x=residuales,bins=N,color='#0504aa')
plt.grid()
plt.xlabel('Residuales')
plt.ylabel('Frecuencia')
plt.title('Distribución de residuales')


# _____________________________ Conclusión ____________________________________
print (" ")
print ("Conlusión")
print (" ")
print ("Gracias al programa en Python en regresión lineal me puedo dar")
print ("cuenta que la felicidad no es un resultado de tener dinero, ya que")
print ("muchos países incluidos en estos datos que no tienen tanto GDP,")
print ("son igual o hasta más felices que países con mas GDP que ellos.")
print ("También noté que el modelo lineal que se usa no es un modelo adecuado")
print ("para este tipo de datos. Esto lo vi porque vi una simetría en la")
print ("gráfica de residuales cosa que no debe de pasar si el modelo lineal")
print ("fuera el correcto para utilizar en estos datos.")

print (" ")
print (" ")



dv = pd.read_csv ("HappyPlanetIndex.csv")


# _____________________________ Probabilidades ________________________________

"""
Un país de Latinoamérica tenga un índice mayor o igual a 5 de felicidad
"""

hp = np.array (dv ["Happiness"])
regions = np. array (dv ["Region"])

count_lat = 0
count_latf = 0
i = 0

for region in regions :
    if region == 1 :
        count_lat += 1
        if hp [i] >= 5 :
            count_latf += 1
    i += 1

prob_lat = count_latf / count_lat

print (f"La probabilidad que un país de Latinoamérica tenga un índice que sea >= a 5 es : {prob_lat}")
print (" ")




"""
Un país del Oeste tenga un índice mayor o igual a 5 de felicidad
"""

hp = np.array (dv ["Happiness"])
regions = np. array (dv ["Region"])

count_west = 0
count_westF = 0
i = 0

for region in regions :
    if region == 2 :
        count_west += 1
        if hp [i] >= 5 :
            count_westF += 1
    i += 1

prob_west = count_westF / count_west

print (f"La probabilidad que un país del Oeste tenga un índice que sea >= a 5 es : {prob_west}")
print (" ")




"""
Un país del Medio Este tenga un índice mayor o igual a 5 de felicidad
"""

hp = np.array (dv ["Happiness"])
regions = np. array (dv ["Region"])

count_middle_east = 0
count_middle_eastF = 0
i = 0

for region in regions :
    if region == 3 :
        count_middle_east += 1
        if hp [i] >= 5 :
            count_middle_eastF += 1
    i += 1

prob_middle_east = count_middle_eastF / count_middle_east

print (f"La probabilidad que un país del Medio Este tenga un índice que sea >= a 5 es : {prob_middle_east}")
print (" ")




"""
Un país del Subharan de Africa tenga un índice mayor o igual a 5 de felicidad
"""

hp = np.array (dv ["Happiness"])
regions = np. array (dv ["Region"])

cont_subsahaf = 0
cont_subsahafF = 0
i = 0

for region in regions :
    if region == 4 :
        cont_subsahaf += 1
        if hp [i] >= 5 :
            cont_subsahafF += 1
    i += 1

prob_subsahaf = cont_subsahafF / cont_subsahaf

print (f"La probabilidad que un país del Subharan de Africa tenga un índice que sea >= a 5 es : {prob_subsahaf}")
print (" ")




"""
Un país de Asia del Sur tenga un índice mayor o igual a 3 de felicidad
"""

hp = np.array (dv ["Happiness"])
regions = np. array (dv ["Region"])

cont_southAsia = 0
cont_southAsiaf = 0
i = 0

for region in regions :
    if region == 5 :
        cont_southAsia += 1
        if hp [i] >= 3 :
            cont_southAsiaf += 1
    i += 1

prob_southAsia = cont_southAsiaf / cont_southAsia

print (f"La probabilidad que un país de Asia del Sur tenga un índice que sea >= a 3 es : {prob_southAsia}")
print (" ")




"""
Un país de Asia del Este tenga un índice mayor o igual a 3 de felicidad
"""

hp = np.array (dv ["Happiness"])
regions = np. array (dv ["Region"])

cont_eastAsia = 0
cont_eastAsiaf = 0
i = 0

for region in regions :
    if region == 6 :
        cont_eastAsia += 1
        if hp [i] >= 3 :
            cont_eastAsiaf += 1
    i += 1

prob_eastAsia = cont_eastAsiaf / cont_eastAsia

print (f"La probabilidad que un país de Asia del Este tenga un índice que sea >= a 3 es : {prob_eastAsia}")
print (" ")




"""
Un país que fue comunista tenga un índice mayor o igual a 3 de felicidad 
"""

hp = np.array (dv ["Happiness"])
regions = np. array (dv ["Region"])

cont_comm = 0
cont_commf = 0
i = 0

for region in regions :
    if region == 7 :
        cont_comm += 1
        if hp [i] >= 3 :
            cont_commf += 1
    i += 1

prob_comm = cont_commf / cont_comm

print (f"La probabilidad que un país que fue comunista tenga un índice que sea >= a 3 es : {prob_comm}")
print (" ")



# _____________________________________ Reseña ________________________________

print (" ")
print ("Reseña")
print (" ")
print ("Lo que me deja este video es que el dinero y la felicidad no van de  ")
print ("la mano. La felicidad el algo que encuentras cuando disfrutas lo que ")
print ("tiene valoras lo que hay a tu discposición. Esto lo podemos ver con ")
print ("los diferentes gráficos presentados en la plática")


