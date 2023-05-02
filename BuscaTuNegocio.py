import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Lectura del archivo de datos
df = pd.read_csv('PosiblesNegociosEmpresas.csv')

# Creación de vectorizador para la columna de descripción
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['descripcion'])

# Creación del modelo Naive Bayes y entrenamiento con la columna 'recomendacion'
model = MultinomialNB()
model.fit(X, df['recomendacion'])

# Solicitud de datos del usuario
gustos = input("¿Cuáles son tus gustos? ")
hobbies = input("¿Qué te gusta hacer? ")
intereses = input("¿Cuáles son tus intereses? ")
estudios = input("¿Qué estudios tienes? ")
capital = float(input("¿Cuál es tu capital en dólares? "))
profesion = input("¿Cuál es tu profesión? ")
experiencia = int(input("¿Cuántos años de experiencia tienes? "))
futuro = input("¿Cómo te ves a futuro? ")
edad = int(input("¿Qué edad tienes? "))

# Preprocesamiento del texto de entrada del usuario
texto = f"{gustos} {hobbies} {intereses} {estudios} {profesion} {futuro}"

# Transformación del texto de entrada en vector de características
X_user = vectorizer.transform([texto])

# Predicción del modelo para el vector de características del usuario
prediccion = model.predict(X_user)

# Impresión de la recomendación de negocio para el usuario
print(f"Basado en tus respuestas, te recomendamos considerar iniciar un negocio de {prediccion[0]}.")
