import sys
import argparse
import os


from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()
    model.forward()
    print(model)
    results = model.train(data='vehicle.yaml',  # 训练参数均可以重新设置
                        epochs=300, 
                        imgsz=640, 
                        workers=8, 
                        batch=16,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov8.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    
