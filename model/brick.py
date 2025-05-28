import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

class BrickDetector:
    def __init__(self,single_model : str = "models/run43_stacked.pt", multi_model : str = "models/run63_figures.onnx", is_video : bool = True):
        self.single_model = YOLO(single_model)
        self.multi_model = YOLO(multi_model)
        self.is_video = is_video

        print(self.multi_model.cfg)
        

    
    
    def detect(self,image: np.ndarray, conf: float = 0.4, model_to_use = 2) -> list[Boxes] | tuple[list[Boxes],list[Boxes]]:
        """

        Args:
            image (np.ndarray): The image to run the detection on
            conf (float, optional): The minimum confidence of the bounding boxes. Defaults to 0.4.
            model_to_use (int, optional): If set to 0, only the single brick model will be used.
                                          If set to 1, only the multi brick model will be used.
                                          If set to 2, both model will be used.
                                    

        Returns:
            list[Boxes] | tuple[list[Boxes],list[Boxes]]:
            The returning bounding boxes, only 1 list be returned if model_to_use is set to 0 or 1, else it will return 2 lists with the results
        """
        single_results = []
        multi_results = []
        if model_to_use == 0 or model_to_use == 2:
            single_results = self.single_model.track(image,stream = True, persist = self.is_video,verbose = False)
        if model_to_use == 1 or model_to_use == 2:
            multi_results = self.multi_model.track(image,stream = True, persist = self.is_video,verbose = False)

        single_bboxes = []
        multi_bboxes = []

        try:
            for result in single_results:
                for box in result.boxes:
                    if box.conf > conf:
                        single_bboxes.append(box)
        except:
            pass

        try:
            for result in multi_results:
                for box in result.boxes:
                    if box.conf > conf:
                        multi_bboxes.append(box)
        except:
            pass

        if model_to_use == 0:
            return single_bboxes
        if model_to_use == 1:
            return multi_bboxes
        if model_to_use == 2:
            return single_bboxes, multi_bboxes
            

def random_color():
    return np.random.randint(0, 255, (3), dtype=np.uint8).tolist()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)

    # Add any transformation here
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


    
def annotate_image(image,bboxes,class_names = ["red","green","blue","yellow","big_penguin","small_penguin","lion","sheep","pig","human"], colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),(0.4,0.4,1),(0.4,1,0.4),(1,1,0.4),(1,1,1),(1,0.5,0.5),(1.0,0.2,0.8)]):
    for box in bboxes:
        # check if confidence is greater than 40 percent
        if box.conf[0] > 0.4:
            # get coordinates
            [x,y,w,h] = box.xywh[0]

            x1 = x-w/2
            x2 = x+w/2
            y1 = y - h/2
            y2 = y + h/2
            # convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # get the class
            cls = int(box.cls[0])

            # get the class name
            class_name = class_names[cls]

            # get the respective color
            color = colors[cls]

            if sum(color) < 10:
                color = (color[0]*255,color[1]*255,color[2]*255)
            # draw the rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # put the class name and confidence on the image
            cv2.putText(image, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def draw_box(image, xywh):
    x,y,w,h = xywh

    x1 = x
    x2 = x+w
    y1 = y
    y2 = y+h

    cv2.rectangle(
        image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=random_color(),thickness=5
    )

def write_file(fn,content):
    with open(fn,"w") as text_file:
        text_file.write(content)

def read_file(fn):
    contents = ""
    with open(fn,"r") as text_file:
        contents = text_file.readlines()
    return contents