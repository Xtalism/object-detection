from ultralytics import YOLO

model = YOLO('Modelos/prueba.pt')

model.export(format = 'onnx')