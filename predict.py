from ultralytics import YOLO

model = YOLO("yolov11_custom.pt")  # renamed in main file

model.predict(source = "0",show = True,save = True, conf = 0.5,line_width = 5,save_crop = False,save_txt = False,
              show_labels = True,show_conf = True,classes = [0,1])