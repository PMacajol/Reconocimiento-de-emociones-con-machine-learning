import os
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

# ===============================
# 1) Parámetros y rutas
# ===============================
DATASET_PATH = r"C:\Users\macaj\Downloads\Banco de Audio"
EMOTIONS = {
    'feliz': 'feliz',
    'triste': 'triste',
    'enojo': 'enojo',
    'miedo': 'miedo'
}
MODEL_FILENAME = "emotion_svm_model.joblib"
SCALER_FILENAME = "emotion_scaler.joblib"
LABELENCODER_FILENAME = "emotion_label_encoder.joblib"

DURATION = 3
SR = 22050


# ===============================
# 2) Extracción de características
# ===============================
def extract_features_from_file(file_path):
    audio, sr = librosa.load(file_path, sr=SR, duration=DURATION, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def extract_features_from_array(audio_array, sr):
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


# ===============================
# 3) Grabación de audio
# ===============================
def record_from_microphone(duration=DURATION, sr=SR):
    print(f"⏺ Grabando {duration} segundos... Presiona Ctrl+C para cancelar.")
    try:
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        return np.squeeze(recording)
    except Exception as e:
        print(f"Error al grabar desde el micrófono: {e}")
        return None


# ===============================
# 4) Entrenamiento del modelo
# ===============================
def train_and_save_model():
    features = []
    labels = []

    for emotion_dir in os.listdir(DATASET_PATH):
        emotion_key = emotion_dir.lower()
        if emotion_key in EMOTIONS:
            emotion_path = os.path.join(DATASET_PATH, emotion_dir)
            for file in os.listdir(emotion_path):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    file_path = os.path.join(emotion_path, file)
                    try:
                        mfcc = extract_features_from_file(file_path)
                        features.append(mfcc)
                        labels.append(EMOTIONS[emotion_key])
                    except Exception as e:
                        print(f"⚠ Error al procesar {file_path}: {e}")

    X = np.array(features)
    y = np.array(labels)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== REPORT DE EVALUACIÓN EN TEST ===")
    print(classification_report(y_test, y_pred))

    # Guardar modelo, escalador y codificador
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    joblib.dump(label_encoder, LABELENCODER_FILENAME)
    print(f"\n✅ Modelo, escalador y codificador guardados como:\n   - {MODEL_FILENAME}\n   - {SCALER_FILENAME}\n   - {LABELENCODER_FILENAME}")

    plot_classification_report(y_test, y_pred, label_encoder)


def plot_classification_report(y_true, y_pred, label_encoder):
    report = classification_report(y_true, y_pred, output_dict=True)
    emotions = [k for k in report.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    f1_scores = [report[e]['f1-score'] for e in emotions]
    emotion_names = label_encoder.inverse_transform([int(e) for e in emotions])

    plt.figure(figsize=(8, 4))
    plt.bar(emotion_names, f1_scores, color='skyblue')
    plt.ylim(0, 1)
    plt.title('Precisión por clase (F1-Score)')
    plt.xlabel('Emoción')
    plt.ylabel('F1-Score')
    plt.tight_layout()
    plt.show()


# ===============================
# 5) Predicción desde micrófono
# ===============================
def predict_emotion_from_microphone():
    if not (os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME) and os.path.exists(LABELENCODER_FILENAME)):
        print("⚠ No se encontraron todos los archivos necesarios. Entrenando primero...")
        train_and_save_model()

    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    label_encoder = joblib.load(LABELENCODER_FILENAME)

    audio = record_from_microphone(duration=DURATION, sr=SR)
    if audio is None:
        print("❌ Falló la grabación. Saliendo.")
        return

    sf.write("grabacion_temp.wav", audio, SR)

    features = extract_features_from_array(audio, SR).reshape(1, -1)
    features_scaled = scaler.transform(features)

    y_prob = model.predict_proba(features_scaled)[0]
    y_pred = model.predict(features_scaled)[0]
    emotion_text = label_encoder.inverse_transform([y_pred])[0]

    print("\n🎤 Resultado de la predicción:")
    for idx, emo in enumerate(label_encoder.classes_):
        print(f"   - {emo}: {y_prob[idx]*100:.2f}%")
    print(f"\n➡ Emoción predicha: {emotion_text.upper()}")


# ===============================
# 6) Menú principal
# ===============================
if __name__ == "__main__":
    print("==============================================")
    print("   Clasificador de Emociones desde Voz")
    print("==============================================\n")
    print("Opciones:")
    print("  1) Entrenar / reentrenar el modelo con el dataset.")
    print("  2) Capturar audio desde micrófono y predecir emoción.")
    print("  0) Salir.\n")

    opcion = input("Selecciona una opción (0/1/2): ").strip()

    if opcion == '1':
        train_and_save_model()
    elif opcion == '2':
        predict_emotion_from_microphone()
    else:
        print("🔚 Saliendo...")

