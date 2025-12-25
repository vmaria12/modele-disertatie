import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tkinter as tk
from tkinter import filedialog
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def select_files():
    root = tk.Tk()
    root.withdraw()  
    file_paths = filedialog.askopenfilenames(
        title="Selectați modelele (.h5, .keras, .mat)",
        filetypes=[("Model Files", "*.h5 *.keras *.mat"), ("All Files", "*.*")]
    )
    return file_paths

def find_testing_dir(base_path):
    if os.path.exists(os.path.join(base_path, 'Testing')):
        return os.path.join(base_path, 'Testing')
        
    for root, dirs, files in os.walk(base_path):
        if 'Testing' in dirs:
            return os.path.join(root, 'Testing')
    return None

def calculate_metrics():
    print("=== Script Calcul Metrici Modele ===")
    
    print("\nVă rugăm să selectați fișierele model din fereastra de dialog care se va deschide...")
    model_paths = select_files()
    
    if not model_paths:
        print("Nu a fost selectat niciun model. Ieșire.")
        return

    base_dataset_path = r'D:\Disertatie\DataSet'
    print(f"Căutare director 'Testing' în: {base_dataset_path} ...")
    test_dir = find_testing_dir(base_dataset_path)
    
    if not test_dir:
        print(f"EROARE: Nu s-a găsit directorul 'Testing' în {base_dataset_path}.")
        print("Verificați structura directorului de date.")
        return
        
    print(f"Director de testare găsit: {test_dir}")

    # Parametri
    IMG_SIZE = 224
    BATCH_SIZE = 32
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("Încărcare imagini de test...")
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )
    
    class_names = list(test_gen.class_indices.keys())
    print(f"Clase identificate: {class_names}")

    results = []

    # 4. Evaluare fiecare model
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\n--------------------------------------------------")
        print(f"Evaluare model: {model_name}")
        
        try:
            model = load_model(model_path)
            
            # Predicție
            print("Generare predicții...")
            Y_pred_probs = model.predict(test_gen, verbose=1)
            y_pred = np.argmax(Y_pred_probs, axis=1)
            y_true = test_gen.classes
            
            # Calcul metrici
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            print(f"Rezultate pentru {model_name}:")
            print(f"  Acuratețe: {acc:.4f}")
            print(f"  Precizie:  {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            
            # Raport detaliat
            print("\nRaport detaliat pe clase:")
            print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
            
            results.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1,
                'Path': model_path
            })
            
        except Exception as e:
            print(f"EROARE la procesarea modelului {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Salvare și Afișare Finală
    if results:
        df = pd.DataFrame(results)
        print("\n==================================================")
        print("REZUMAT FINAL")
        print("==================================================")
        print(df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']])
        
        # Salvare în CSV în același director cu scriptul
        output_csv = os.path.join(os.path.dirname(__file__), 'rezultate_metrice_comparative.csv')
        df.to_csv(output_csv, index=False)
        print(f"\nRezultatele detaliate au fost salvate în: {output_csv}")
    else:
        print("\nNu s-au generat rezultate (posibile erori sau niciun model selectat).")

if __name__ == "__main__":
    calculate_metrics()