import torch
import cv2
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import time 

torch.cuda.set_device(0)

# Load a COCO-pretrained YOLO-NAS-s model
#model = NAS("yolo_nas_l.pt")
#model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.engine')
#model = models.get("yolo_nas_l", pretrained_weights="coco")
model = YOLO("18_10_23-modely8m.pt")
# model = YOLO("18_10_23-modely8m.engine")
print("modelo cargado")

#print(model)


#model.to("cuda")
#model.eval()
# Abre el archivo de video
video_path = "Videos_prueba/videoGrua.mp4"  # Reemplaza 'video.mp4' con la ruta de tu propio archivo de video
# cap = cv2.VideoCapture(video_path)
results = model(video_path, show=True, conf=0.7)

# Verifica si el video se abrió correctamente
# if not cap.isOpened():
#     print("Error al abrir el video")
#     exit()

# while True:
#     # Lee un cuadro del video
#     ret, image = cap.read()

#     # Si no se pudo leer un cuadro, sal del bucle
#     if not ret:
#         break
 
#     inicio=time.time()
#     result = model(image, show=True)
#     fin=time.time()
#     print('Time procesing: '+str(fin-inicio))
    
#     #result = model(image,stream=True)

    
#     # Muestra el cuadro en una ventana
#     # cv2.imshow("Video", result)


#     # ## Espera una pequeña cantidad de tiempo y verifica si se presionó la tecla 'q' para salir
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break

# # Libera el objeto de captura y cierra la ventana
# cap.release()
# cv2.destroyAllWindows()

