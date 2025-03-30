import pandas as pd
import numpy as np
from collections import defaultdict
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Descargar recursos de NLTK
print("Descargando recursos necesarios de NLTK...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
print("Recursos descargados correctamente.\n")

class Preprocesador:
    def __init__(self):
        """Inicializa el preprocesador cargando las stopwords en inglés
        
        Atributos:
            stopwords (set): Conjunto de palabras vacías (stopwords) en inglés 
                            que serán filtradas durante el preprocesamiento
        """
        self.stopwords = set(stopwords.words('english'))  # Carga stopwords de NLTK
    
    def preprocesar(self, texto):
        
        # Verificar que el input sea texto válido
        if not isinstance(texto, str):
            return []  # Devuelve lista vacía si no es string
        
        # Normalización: convertir todo a minúsculas
        texto = texto.lower()
        
        # Eliminar signos de puntuación (todo excepto caracteres de palabra y espacios)
        texto = re.sub(r'[^\w\s]', '', texto)
        
        #Tokenización: dividir el texto en palabras individuales
        tokens = word_tokenize(texto)
        
        # Eliminar stopwords (palabras comunes sin significado) y palabras muy cortas (longitud <= 2 caracteres)
        tokens = [palabra for palabra in tokens 
                 if palabra not in self.stopwords and len(palabra) > 2]
        
        return tokens  # Devuelve lista de tokens limpios

# Teorema de Bayes
class BayesSpam:
    def __init__(self):
        self.P_spam = None       # P(Spam)
        self.P_ham = None        # P(No Spam)
        self.P_palabra_spam = {} # P(Palabra|Spam)
        self.P_palabra_ham = {}  # P(Palabra|No Spam)
        self.vocabulario = set()
    
    def entrenar(self, correos, etiquetas):
        """Entrena el modelo con los correos y etiquetas dados"""
        print("\nTeorema de Bayes...")
        
        # Calcular probabilidades a priori
        n_spam = sum(etiquetas)
        n_total = len(etiquetas)
        self.P_spam = n_spam / n_total
        self.P_ham = 1 - self.P_spam
        
        print(f"Probabilidad a priori - Spam: {self.P_spam:.4f}")
        print(f"Probabilidad a priori - No Spam: {self.P_ham:.4f}")
        
        # Contar frecuencias de palabras
        frec_spam = defaultdict(int)
        frec_ham = defaultdict(int)
        total_palabras_spam = 0
        total_palabras_ham = 0
        
        for correo, es_spam in zip(correos, etiquetas):
            for palabra in correo:
                self.vocabulario.add(palabra)
                if es_spam:
                    frec_spam[palabra] += 1
                    total_palabras_spam += 1
                else:
                    frec_ham[palabra] += 1
                    total_palabras_ham += 1
        
        print(f"Tamaño del vocabulario: {len(self.vocabulario)} palabras")
        print(f"Palabras en spam: {total_palabras_spam}")
        print(f"Palabras en no spam: {total_palabras_ham}\n")
        
        # Calcular probabilidades condicionales con suavizado Laplace
        for palabra in self.vocabulario:
            # Probabilidad de que la palabra aparezca en un mensaje SPAM (P(palabra|spam))
            self.P_palabra_spam[palabra] = (frec_spam.get(palabra, 0) + 1) / (total_palabras_spam + len(self.vocabulario))
            # Probabilidad de que la palabra aparezca en un mensaje NO SPAM (P(palabra|ham))
            self.P_palabra_ham[palabra] = (frec_ham.get(palabra, 0) + 1) / (total_palabras_ham + len(self.vocabulario))
    
    def predecir(self, correo):
        """Predice si un correo es spam (1) o no spam (0)"""
        # Inicializa con la probabilidad a priori en escala logarítmica
        log_P_spam = np.log(self.P_spam)
        log_P_ham = np.log(self.P_ham)
        
        # Acumula la evidencia de cada palabra del correo
        for palabra in correo:
            if palabra in self.vocabulario:
                # Suma el logaritmo de las probabilidades condicionales
                log_P_spam += np.log(self.P_palabra_spam[palabra])
                log_P_ham += np.log(self.P_palabra_ham[palabra])
                
        # Decide la clase con mayor probabilidad logarítmica acumulada
        return 1 if log_P_spam > log_P_ham else 0

# Impresion mejorada del reporte
def reporte_en_espanol(y_true, y_pred):
    reporte = classification_report(y_true, y_pred, target_names=["NO SPAM", "SPAM"], output_dict=True)
    
    print("\nRESULTADOS DEL MODELO:")
    print("="*50)
    print(f"{'CLASIFICACIÓN':^50}")
    print("="*50)
    print(f"{' ':20}{'Precisión':<10}{'Sensibilidad':<12}{'F1-score':<10}{'Soporte':<8}")
    print("-"*50)
    
    # Datos para NO SPAM
    print(f"{'NO SPAM':<20}", end="")
    print(f"{reporte['NO SPAM']['precision']:<10.2f}", end="")
    print(f"{reporte['NO SPAM']['recall']:<12.2f}", end="")
    print(f"{reporte['NO SPAM']['f1-score']:<10.2f}", end="")
    print(f"{int(reporte['NO SPAM']['support']):<8}")
    
    # Datos para SPAM
    print(f"{'SPAM':<20}", end="")
    print(f"{reporte['SPAM']['precision']:<10.2f}", end="")
    print(f"{reporte['SPAM']['recall']:<12.2f}", end="")
    print(f"{reporte['SPAM']['f1-score']:<10.2f}", end="")
    print(f"{int(reporte['SPAM']['support']):<8}")
    
    print("-"*50)
    
    # Promedios
    print(f"{'Promedio macro':<20}", end="")
    print(f"{reporte['macro avg']['precision']:<10.2f}", end="")
    print(f"{reporte['macro avg']['recall']:<12.2f}", end="")
    print(f"{reporte['macro avg']['f1-score']:<10.2f}", end="")
    print(f"{int(reporte['macro avg']['support']):<8}")
    
    print(f"{'Promedio ponderado':<20}", end="")
    print(f"{reporte['weighted avg']['precision']:<10.2f}", end="")
    print(f"{reporte['weighted avg']['recall']:<12.2f}", end="")
    print(f"{reporte['weighted avg']['f1-score']:<10.2f}", end="")
    print(f"{int(reporte['weighted avg']['support']):<8}")
    
    print("="*50)
    print(f"{'Exactitud global del modelo:':<30}{accuracy_score(y_true, y_pred):.4f}")
    print("="*50)

# Cargar datos
print("\nCargando y preparando los datos...")
try:
    data = pd.read_csv("Spamrial.csv", encoding="latin-1")
    data = data[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    data["label"] = data["label"].map({"spam": 1, "ham": 0})
    print(f"Datos cargados: {len(data)} mensajes (Spam: {sum(data['label'])}, No spam: {len(data)-sum(data['label'])})")
except Exception as e:
    print(f"Error al cargar datos: {str(e)}")
    exit()

# Preprocesar textos
print("\nPreprocesando textos...")
prep = Preprocesador()
correos_procesados = []
for texto in data["text"]:
    try:
        procesado = prep.preprocesar(texto)
        correos_procesados.append(procesado)
    except Exception as e:
        print(f"Error procesando texto: {str(e)}")
        correos_procesados.append([])

etiquetas = data["label"].tolist()

# Dividir en entrenamiento y prueba
print("\nDividiendo datos en entrenamiento (80%) y prueba (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    correos_procesados, etiquetas, test_size=0.2, random_state=42
)
print(f"Datos de entrenamiento: {len(X_train)} mensajes")
print(f"Datos de prueba: {len(X_test)} mensajes")

# Entrenar modelo
modelo = BayesSpam()
modelo.entrenar(X_train, y_train)

# Evaluar modelo
print("\nEvaluando modelo con datos de prueba...")
y_pred = [modelo.predecir(correo) for correo in X_test]

# Mostrar reporte en español
reporte_en_espanol(y_test, y_pred)

# Ejemplo de predicción
test_messages = [
    "Free entry to win a prize! Call now!",  # SPAM
    "Hey, can we meet tomorrow for lunch?",  # NO SPAM
    "Congratulations! You've won a $1000 gift card",  # SPAM
    "Please send me the report by EOD",  # NO SPAM
    "Earn money fast from house",  # SPAM
    "Meeting tomorrow at 10am"  # NO SPAM
]

print("\nPREDICCIONES DE EJEMPLO:")
print("="*50)
for msg in test_messages:
    procesado = prep.preprocesar(msg)
    pred = modelo.predecir(procesado)
    print(f"Mensaje: '{msg[:50]}...'")
    print(f"Clasificación: {'SPAM' if pred == 1 else 'NO SPAM'}")
    print("-"*50)