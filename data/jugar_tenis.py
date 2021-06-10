# Ampliacion de Inteligencia Artificial.
# Grado en Ingenieria Informatica - Tecnologias Informaticas
# jugar_tenis.py
# Ejemplo visto en clase (ver diapositivas)

# atributos=[('Cielo',['Soleado','Nublado','Lluvia']),
#            ('Temperatura',['Alta','Baja','Suave']),
#            ('Humedad',['Alta','Normal']),
#            ('Viento',['Debil','Fuerte'])]

# atributo_clasificacion='Jugar Tenis'
# clases=['si','no']

import numpy as np


X_tenis=np.array([['Soleado' , 'Alta'        , 'Alta'    , 'Debil'], 
                  ['Soleado' , 'Alta'        , 'Alta'    , 'Fuerte'], 
                  ['Nublado' , 'Alta'        , 'Alta'    , 'Debil'],  
                  ['Lluvia'  , 'Suave'       , 'Alta'    , 'Debil'],  
                  ['Lluvia'  , 'Baja'        , 'Normal'  , 'Debil' ], 
                  ['Lluvia'  , 'Baja'        , 'Normal'  , 'Fuerte'], 
                  ['Nublado' , 'Baja'        , 'Normal'  , 'Fuerte'], 
                  ['Soleado' , 'Suave'       , 'Alta'    , 'Debil'],  
                  ['Soleado' , 'Baja'        , 'Normal'  , 'Debil'],  
                  ['Lluvia'  , 'Suave'       , 'Normal'  , 'Debil'],  
                  ['Soleado' , 'Suave'       , 'Normal'  , 'Fuerte'], 
                  ['Nublado' , 'Suave'       , 'Alta'    , 'Fuerte'], 
                  ['Nublado' , 'Alta'        , 'Normal'  , 'Debil'],  
                  ['Lluvia'  , 'Suave'       , 'Alta'    , 'Fuerte']])

y_tenis=np.array(['no',
                  'no',      
                  'si',
                  'si',
                  'si',
                  'no',
                  'si',
                  'no',
                  'si',
                  'si',
                  'si',
                  'si',
                  'si',
                  'no'])
