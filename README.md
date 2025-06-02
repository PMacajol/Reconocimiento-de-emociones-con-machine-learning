# Reconocimiento-de-emociones-con-machine-learning

Documentacion 

 
SECCIÓN 1: Librerías
import os  # Para trabajar con archivos y directorios (listar carpetas, construir rutas)
import librosa  # Para cargar y analizar archivos de audio (especialmente en tareas de Machine Learning)
import numpy as np  # Librería para cálculos numéricos y arreglos (vectores y matrices)
import sounddevice as sd  # Para grabar audio directamente desde el micrófono
import soundfile as sf  # Para guardar audio grabado en un archivo .wav
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y prueba
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Para codificar etiquetas y escalar datos numéricos
from sklearn.svm import SVC  # El modelo SVM (Support Vector Machine) para clasificación
from sklearn.metrics import classification_report  # Para obtener métricas del modelo (precisión, recall, F1)
import matplotlib.pyplot as plt  # Para graficar los resultados
import joblib  # Para guardar y cargar el modelo ya entrenado
________________________________________
SECCIÓN 2: Parámetros y rutas
DATASET_PATH = r"C:\Users\macaj\Downloads\Banco de Audio"  # Ruta donde están las carpetas con audios
EMOTIONS = {
    'feliz': 'feliz',
    'triste': 'triste',
    'enojo': 'enojo',
    'miedo': 'miedo'
}
Diccionario para mapear nombres de carpetas a emociones. Las carpetas deben llamarse “feliz”, “triste”, etc.
MODEL_FILENAME = "emotion_svm_model.joblib"  # Nombre del archivo donde se guarda el modelo entrenado
SCALER_FILENAME = "emotion_scaler.joblib"  # Archivo donde se guarda el escalador (StandardScaler)
LABELENCODER_FILENAME = "emotion_label_encoder.joblib"  # Archivo del codificador de etiquetas (LabelEncoder)

DURATION = 3  # Duración del audio a grabar o cargar (en segundos)
SR = 22050  # Frecuencia de muestreo del audio (típico en análisis de voz)
________________________________________
 SECCIÓN 3: Extracción de características
def extract_features_from_file(file_path):
    audio, sr = librosa.load(file_path, sr=SR, duration=DURATION, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)
•	Carga un archivo de audio.
•	Calcula los coeficientes MFCC (Mel Frequency Cepstral Coefficients), que resumen cómo suena la voz.
•	Retorna el promedio de esos coeficientes como un vector.
def extract_features_from_array(audio_array, sr):
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)
•	Igual que la función anterior, pero trabaja con audio en forma de array (grabado en tiempo real).
________________________________________
SECCIÓN 4: Grabación desde micrófono
def record_from_microphone(duration=DURATION, sr=SR):
    print(f"⏺ Grabando {duration} segundos... Presiona Ctrl+C para cancelar.")
    try:
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        return np.squeeze(recording)
    except Exception as e:
        print(f"Error al grabar desde el micrófono: {e}")
        return None
•	Graba audio durante 3 segundos.
•	sd.rec graba y sd.wait() espera a que termine.
•	np.squeeze convierte el array de 2D a 1D.
________________________________________
SECCIÓN 5: Entrenamiento del modelo
def train_and_save_model():
    features = []
    labels = []
•	features: lista de vectores MFCC.
•	labels: lista con las emociones correspondientes.
    for emotion_dir in os.listdir(DATASET_PATH):
        emotion_key = emotion_dir.lower()
        if emotion_key in EMOTIONS:
            emotion_path = os.path.join(DATASET_PATH, emotion_dir)
•	Recorre las carpetas del dataset.
•	Si el nombre de la carpeta está en el diccionario, entra.
            for file in os.listdir(emotion_path):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    file_path = os.path.join(emotion_path, file)
•	Busca archivos .wav o .mp3 dentro de cada carpeta de emoción.
                    try:
                        mfcc = extract_features_from_file(file_path)
                        features.append(mfcc)
                        labels.append(EMOTIONS[emotion_key])
                    except Exception as e:
                        print(f"⚠ Error al procesar {file_path}: {e}")
•	Extrae los MFCC y los guarda junto con su etiqueta.
•	Si hay error, lo muestra.
    X = np.array(features)
    y = np.array(labels)
•	Convierte listas a arreglos NumPy.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
•	Convierte las etiquetas (‘feliz’, ‘enojo’, etc.) en números.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
•	Normaliza los datos para que todas las características tengan media 0 y varianza 1.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
•	Divide el dataset: 80% para entrenar, 20% para probar.
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
•	Crea y entrena un clasificador SVM lineal.
    y_pred = model.predict(X_test)
    print("\n=== REPORT DE EVALUACIÓN EN TEST ===")
    print(classification_report(y_test, y_pred))
•	Predice y muestra el rendimiento del modelo.
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    joblib.dump(label_encoder, LABELENCODER_FILENAME)
•	Guarda el modelo, el escalador y el codificador de etiquetas en disco.
    plot_classification_report(y_test, y_pred, label_encoder)
•	Muestra una gráfica de desempeño por emoción.
________________________________________
SECCIÓN 6: Gráfico de F1-Score
def plot_classification_report(y_true, y_pred, label_encoder):
    report = classification_report(y_true, y_pred, output_dict=True)
    emotions = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    f1_scores = [report[e]['f1-score'] for e in emotions]
    emotion_names = label_encoder.inverse_transform([int(e) for e in emotions])
•	Calcula el F1-score por emoción.
•	Convierte los índices numéricos a etiquetas de emoción.
    plt.figure(figsize=(8, 4))
    plt.bar(emotion_names, f1_scores, color='skyblue')
    plt.ylim(0, 1)
    plt.title('Precisión por clase (F1-Score)')
    plt.xlabel('Emoción')
    plt.ylabel('F1-Score')
    plt.tight_layout()
    plt.show()
•	Crea y muestra una gráfica de barras del rendimiento por clase.
________________________________________
SECCIÓN 7: Predicción en tiempo real
def predict_emotion_from_microphone():
    if not (os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME) and os.path.exists(LABELENCODER_FILENAME)):
        print("⚠ No se encontraron todos los archivos necesarios. Entrenando primero...")
        train_and_save_model()
•	Si no existen los archivos del modelo, lo entrena.
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    label_encoder = joblib.load(LABELENCODER_FILENAME)
•	Carga el modelo, el escalador y el codificador.
python
CopiarEditar
    audio = record_from_microphone(duration=DURATION, sr=SR)
    if audio is None:
        print("❌ Falló la grabación. Saliendo.")
        return
•	Graba audio desde el micrófono.
    sf.write("grabacion_temp.wav", audio, SR)
•	Guarda temporalmente el audio grabado.
    features = extract_features_from_array(audio, SR).reshape(1, -1)
    features_scaled = scaler.transform(features)
•	Extrae y normaliza los MFCC del audio grabado.
    y_prob = model.predict_proba(features_scaled)[0]
    y_pred = model.predict(features_scaled)[0]
    emotion_text = label_encoder.inverse_transform([y_pred])[0]
•	Predice la emoción y convierte el número a texto.
    print("\n🎤 Resultado de la predicción:")
    for idx, emo in enumerate(label_encoder.classes_):
        print(f"   - {emo}: {y_prob[idx]*100:.2f}%")
    print(f"\n➡ Emoción predicha: {emotion_text.upper()}")
•	Muestra las probabilidades de cada emoción y la predicción final.
________________________________________
SECCIÓN 8: Menú principal
if __name__ == "__main__":
•	Solo se ejecuta si el archivo se corre directamente (no si se importa como módulo).
    print("==============================================")
    print("   Clasificador de Emociones desde Voz")
    print("==============================================\n")
    print("Opciones:")
    print("  1) Entrenar / reentrenar el modelo con el dataset.")
    print("  2) Capturar audio desde micrófono y predecir emoción.")
    print("  0) Salir.\n")
•	Imprime el menú de opciones para el usuario.
    opcion = input("Selecciona una opción (0/1/2): ").strip()
•	Espera que el usuario seleccione una opción.
    if opcion == '1':
        train_and_save_model()
    elif opcion == '2':
        predict_emotion_from_microphone()
    else:
        print("🔚 Saliendo...")
•	Ejecuta la función correspondiente o sale del programa.










