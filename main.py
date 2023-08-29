from ultralytics import YOLO

model = YOLO("C:/Users/Ngo Van Vinh/Downloads/runs/detect/train5/weights/last.pt")
if __name__ == '__main__':
    results = model("test8.jpg", save =True)
    # model.train(data="coco1289.yaml", epochs=20, batch=16, device="cpu")