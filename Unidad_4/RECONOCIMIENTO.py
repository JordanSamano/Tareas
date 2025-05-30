import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Emociones")
        self.root.geometry("1000x700")  # Aumentado para el cuadro de texto
        
        # Variables
        self.dataset_path = r"C:\Users\jorda\Desktop\Sem 8\I.A\DATAPROCESADO"
        self.method = tk.StringVar(value="LBPH")
        self.training = False
        self.emotion_recognizer = None
        self.label_dict = {}
        self.current_emotion = tk.StringVar(value="No detectado")
        
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Panel izquierdo (controles)
        control_frame = tk.Frame(self.root, width=300, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Título
        tk.Label(control_frame, text="Configuración", font=("Arial", 14), bg="#f0f0f0").pack(pady=10)
        
        # Método de reconocimiento
        tk.Label(control_frame, text="Método:", bg="#f0f0f0").pack()
        tk.Radiobutton(control_frame, text="LBPH", variable=self.method, value="LBPH", bg="#f0f0f0").pack(anchor=tk.W)
        tk.Radiobutton(control_frame, text="EigenFaces", variable=self.method, value="EigenFaces", bg="#f0f0f0").pack(anchor=tk.W)
        tk.Radiobutton(control_frame, text="FisherFaces", variable=self.method, value="FisherFaces", bg="#f0f0f0").pack(anchor=tk.W)
        
        # Botón de entrenamiento
        tk.Button(control_frame, text="Entrenar Modelo", command=self.start_training, bg="#4CAF50", fg="white").pack(pady=20, fill=tk.X)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=250, mode='determinate')
        self.progress.pack(pady=10)
        self.progress_label = tk.Label(control_frame, text="", bg="#f0f0f0")
        self.progress_label.pack()
        
        # Panel derecho (webcam y emociones)
        camera_emotion_frame = tk.Frame(self.root)
        camera_emotion_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Frame para la webcam
        self.camera_frame = tk.Frame(camera_emotion_frame, bg="black")
        self.camera_frame.pack(expand=True, fill=tk.BOTH)
        
        # Etiqueta para mostrar la webcam
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(expand=True, fill=tk.BOTH)
        
        # Frame para mostrar la emoción detectada
        emotion_frame = tk.Frame(camera_emotion_frame, bg="#f0f0f0", height=50)
        emotion_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Etiqueta de título
        tk.Label(emotion_frame, text="Emoción detectada:", font=("Arial", 12), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
        
        # Cuadro de texto para mostrar la emoción
        emotion_display = tk.Label(emotion_frame, textvariable=self.current_emotion, 
                                 font=("Arial", 14, "bold"), bg="white", relief=tk.SUNKEN, 
                                 width=20, anchor=tk.CENTER)
        emotion_display.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        
        # Etiqueta de estado
        self.status_label = tk.Label(self.root, text="Preparado", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def start_training(self):
        if self.training:
            return
            
        self.training = True
        self.status_label.config(text="Entrenando modelo...")
        
        # Detener la cámara si está activa
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        # Usar un hilo separado para el entrenamiento
        training_thread = threading.Thread(target=self.train_model)
        training_thread.daemon = True
        training_thread.start()
    
    def update_progress(self, current, total, message):
        self.progress['value'] = (current / total) * 100
        self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def load_dataset(self):
        faces = []
        labels = []
        label_dict = {}
        current_label = 0
        
        # Mapeo de nombres de emociones en inglés a español
        emotion_translation = {
            'angry': 'Enojo',
            'disgust': 'Disgusto',
            'fear': 'Miedo',
            'happy': 'Felicidad',
            'neutral': 'Neutral',
            'sad': 'Tristeza',
            'surprise': 'Sorpresa'
        }
        
        # Procesar tanto train como test
        for dataset_type in ['train', 'test']:
            dataset_type_path = os.path.join(self.dataset_path, dataset_type)
            
            if not os.path.exists(dataset_type_path):
                continue
                
            for emotion_folder in os.listdir(dataset_type_path):
                emotion_path = os.path.join(dataset_type_path, emotion_folder)
                
                if os.path.isdir(emotion_path):
                    # Usar el nombre traducido si existe
                    emotion_name = emotion_translation.get(emotion_folder.lower(), emotion_folder)
                    
                    if current_label not in label_dict:
                        label_dict[current_label] = emotion_name
                    
                    image_files = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    total_images = len(image_files)
                    
                    for i, image_name in enumerate(image_files):
                        image_path = os.path.join(emotion_path, image_name)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        
                        if image is not None:
                            image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
                            faces.append(image)
                            labels.append(current_label)
                            
                            # Actualizar progreso cada 10 imágenes
                            if i % 10 == 0:
                                self.update_progress(
                                    len(faces), 
                                    total_images, 
                                    f"Procesando {emotion_name}: {i+1}/{total_images}"
                                )
                    
                    current_label += 1
        
        return np.array(faces), np.array(labels), label_dict
    
    def train_model(self):
        try:
            # Cargar dataset
            self.update_progress(0, 100, "Cargando dataset...")
            faces, labels, label_dict = self.load_dataset()
            
            if len(faces) == 0:
                messagebox.showerror("Error", "No se encontraron imágenes válidas en el dataset")
                self.training = False
                return
                
            self.label_dict = label_dict
            self.update_progress(0, 100, f"Dataset cargado: {len(faces)} imágenes, {len(label_dict)} emociones")
            
            # Entrenar modelo
            method = self.method.get()
            self.update_progress(0, 100, f"Entrenando modelo {method}...")
            
            if method == 'EigenFaces':
                recognizer = cv2.face.EigenFaceRecognizer_create()
            elif method == 'FisherFaces':
                recognizer = cv2.face.FisherFaceRecognizer_create()
            else:  # LBPH
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            recognizer.train(faces, labels)
            recognizer.save(f'modelo{method}.xml')
            self.emotion_recognizer = recognizer
            
            # Iniciar reconocimiento por webcam
            self.update_progress(100, 100, "Entrenamiento completado!")
            self.status_label.config(text="Modelo entrenado. Iniciando cámara...")
            self.start_webcam()
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
        finally:
            self.training = False
    
    def start_webcam(self):
        # Detener cámara si ya está abierta
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        # Iniciar nueva captura
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo acceder a la webcam")
            return
            
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.status_label.config(text="Webcam activa - Presiona ESC para salir")
        self.update_webcam()
    
    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = self.face_classifier.detectMultiScale(gray, 1.3, 5)
            
            emotion_detected = "No detectado"
            confidence_level = 0
            
            for (x, y, w, h) in faces_detected:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_CUBIC)
                
                if self.emotion_recognizer is not None:
                    label, confidence = self.emotion_recognizer.predict(face_roi)
                    emotion_name = self.label_dict.get(label, "Desconocido")
                    
                    # Configurar umbrales
                    if self.method.get() == 'EigenFaces':
                        threshold = 5700
                    elif self.method.get() == 'FisherFaces':
                        threshold = 500
                    else:  # LBPH
                        threshold = 60
                    
                    if emotion_name != "Desconocido" and confidence < threshold:
                        color = (0, 255, 0)  # Verde
                        text = f"{emotion_name} ({confidence:.2f})"
                        emotion_detected = emotion_name
                        confidence_level = confidence
                    else:
                        color = (0, 0, 255)  # Rojo
                        text = "No identificado"
                        emotion_detected = "No identificado"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Actualizar el cuadro de texto de emoción
            if emotion_detected != "No detectado":
                self.current_emotion.set(f"{emotion_detected} ({confidence_level:.1f})")
            else:
                self.current_emotion.set(emotion_detected)
            
            # Mostrar frame en la interfaz
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            
            self.camera_label.img = img  # Mantener referencia
            self.camera_label.config(image=img)
        
        # Continuar actualizando
        self.root.after(10, self.update_webcam)
    
    def on_closing(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # Verificar que opencv-contrib-python esté instalado
    try:
        cv2.face
    except AttributeError:
        print("Error: opencv-contrib-python no está instalado.")
        print("Ejecuta: pip install opencv-contrib-python")
        exit()
    
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()