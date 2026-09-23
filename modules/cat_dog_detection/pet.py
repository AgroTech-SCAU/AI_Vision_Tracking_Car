from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="data/openimages_cat_dog_yolo/data.yaml",

    epochs=150,
    imgsz=640,
    batch=8,
    device=0,

    mosaic=0.5,
    close_mosaic=10,

    mixup=0.0,

    scale=0.45,
    translate=0.1,
    fliplr=0.5,

    cos_lr=True,
    lr0=0.001,
    lrf=0.05,

    patience=30,
    save=True,
    plots=True,

    name="openimages_cat_dog"
)

print("完成!")