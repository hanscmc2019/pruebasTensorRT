import torch
import cv2
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import time 
import tensorrt

torch.cuda.set_device(0)
tiempos_ejecucion = []
# Load a COCO-pretrained YOLO-NAS-s model
#model = NAS("yolo_nas_l.pt")
#model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.engine')
#model = models.get("yolo_nas_l", pretrained_weights="coco")
# model = YOLO("model_L_22_09.pt")
# model = YOLO("18_10_23-modely8m_openvino_model/")
model = YOLO("model_L_22_09.engine")
print("modelo cargado")
#print(model)


# model.to("cuda")
# model.eval()
# Abre el archivo de video
video_path = "Videos_prueba/videoGrua.mp4"  # Reemplaza 'video.mp4' con la ruta de tu propio archivo de video
cap = cv2.VideoCapture(video_path)

# Verifica si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

while True:
    # Lee un cuadro del video
    ret, image = cap.read()

    # Si no se pudo leer un cuadro, sal del bucle
    if not ret:
        break
 
    inicio=time.time()
    result = model(image)
    fin=time.time()
    tiempo_ejecucion=fin-inicio
    print('Time procesing: '+str(tiempo_ejecucion))
    tiempos_ejecucion.append(tiempo_ejecucion)
    #result = model(image,stream=True)

    
    # Muestra el cuadro en una ventana
    cv2.imshow("Video", result[0].plot())


    # ## Espera una pequeña cantidad de tiempo y verifica si se presionó la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera el objeto de captura y cierra la ventana
cap.release()
cv2.destroyAllWindows()
tiempo_promedio = sum(tiempos_ejecucion) / len(tiempos_ejecucion)
print("Tiempo promedio:", tiempo_promedio)