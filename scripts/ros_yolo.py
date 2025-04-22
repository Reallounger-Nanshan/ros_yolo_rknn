#!/home/ubuntu/anaconda3/envs/toolkit2/bin/python3.10
import cv2
import time
import numpy as np
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rknn.api import RKNN
from copy import copy


class LetterBoxInfo():
    def __init__(self, shape, new_shape, ratio, dw, dh, pad_color) -> None:
        self.origin_shape = shape
        self.new_shape = new_shape
        self.ratio = ratio
        self.dw = dw 
        self.dh = dh
        self.pad_color = pad_color
        

# Create RKNN model container
class RKNNModelContainer():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn 


    def Run(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result


# YOLOv5 main program
class Yolo():
    def __init__(self):
        self.model_path = "../weights/yolov5s.onnx"
        self.target = "rk3588"
        self.device_id = "None"
        self.img_show = True
        self.img_size = (640, 640)
        self.obj_thresh = 0.25
        self.nms_thresh = 0.45
        self.cls = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
                    "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
                    "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
                    "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
                    "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
                    "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")
        self.coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        
        with open("../anchors/anchors_yolov5.txt", "r") as f:
            values = [float(_v) for _v in f.readlines()]
            self.anchors = np.array(values).reshape(3,-1,2).tolist()
        self.model = RKNNModelContainer(self.model_path, self.target, self.device_id)
        self.letter_box_info_list = []
        
        self.detect_pub = rospy.Publisher('/ros_yolo_rknn/target_position', Odometry, queue_size = 10)


    def FilterBoxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= self.obj_thresh)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores


    def NMSBoxes(self, boxes, scores):
        """
        Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


    def BoxProcess(self, position, anchor):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.img_size[1]//grid_h, self.img_size[0]//grid_w]).reshape(1,2,1,1)

        col = col.repeat(len(anchor), axis=0)
        row = row.repeat(len(anchor), axis=0)
        anchor = np.array(anchor)
        anchor = anchor.reshape(*anchor.shape, 1, 1)

        box_xy = position[:,:2,:,:]*2 - 0.5
        box_wh = pow(position[:,2:4,:,:]*2, 2) * anchor

        box_xy += grid
        box_xy *= stride
        box = np.concatenate((box_xy, box_wh), axis=1)

        # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
        xyxy = np.copy(box)
        xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
        xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
        xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
        xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

        return xyxy


    def SpFlatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)


    def PostProcess(self, input_data):
        boxes, scores, classes_conf = [], [], []
        # 1*255*h*w -> 3*85*h*w
        input_data = [_in.reshape([len(self.anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
        for i in range(len(input_data)):
            boxes.append(self.BoxProcess(input_data[i][:,:4,:,:], self.anchors[i]))
            scores.append(input_data[i][:,4:5,:,:])
            classes_conf.append(input_data[i][:,5:,:,:])

        boxes = [self.SpFlatten(_v) for _v in boxes]
        classes_conf = [self.SpFlatten(_v) for _v in classes_conf]
        scores = [self.SpFlatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.FilterBoxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []

        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.NMSBoxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores


    def LetterBox(self, im, new_shape = (640, 640), pad_color = (0, 0, 0)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation = cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value = pad_color)  # add border
        
        self.letter_box_info_list.append(LetterBoxInfo(shape, new_shape, r, dw, dh, pad_color))
        
        return im


    def Loadimg(self, im, im_size):
        im0 = im
        im = self.LetterBox(im0, new_shape = im_size)
        im = img.transpose((2, 0, 1))[::-1]  # BGR to RGB
        im = np.ascontiguousarray(im)
        return im, im0


    def GetRealBox(self, box):
        bbox = copy(box)
        
        # unletter_box result
        bbox[:,0] -= self.letter_box_info_list[-1].dw
        bbox[:,0] /= self.letter_box_info_list[-1].ratio
        bbox[:,0] = np.clip(bbox[:,0], 0, self.letter_box_info_list[-1].origin_shape[1])

        bbox[:,1] -= self.letter_box_info_list[-1].dh
        bbox[:,1] /= self.letter_box_info_list[-1].ratio
        bbox[:,1] = np.clip(bbox[:,1], 0, self.letter_box_info_list[-1].origin_shape[0])

        bbox[:,2] -= self.letter_box_info_list[-1].dw
        bbox[:,2] /= self.letter_box_info_list[-1].ratio
        bbox[:,2] = np.clip(bbox[:,2], 0, self.letter_box_info_list[-1].origin_shape[1])

        bbox[:,3] -= self.letter_box_info_list[-1].dh
        bbox[:,3] /= self.letter_box_info_list[-1].ratio
        bbox[:,3] = np.clip(bbox[:,3], 0, self.letter_box_info_list[-1].origin_shape[0])
        
        return bbox


    def Run(self, image_msg):
        start_time = time.time()
        
        self.letter_box_info_list = []
        
        ros_img = np.frombuffer(image.data, dtype = np.uint8).reshape(image.height, image.width, -1)
        img, img0 = self.Loadimg(ros_img, self.img_size)

        outputs = self.model.Run([np.expand_dims(img, axis = 0)])
        boxes, classes, scores = self.PostProcess(outputs)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print("\n", "%.2f"%fps, "FPS\n")

        if self.img_show:
            drawing = img0.copy()

        if boxes is not None:
            for box, score, cl in zip(self.GetRealBox(boxes), scores, classes):
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = [int(_b) for _b in box]
                print("%s @ (%d %d %d %d) %.3f" % (cls[cl], top_left_x, top_left_y, right, bottom_right_x, bottom_right_y))
                
                target_position = Odometry()
                target_position.pose.pose.position.x = (top_left_x + bottom_right_x) / 2
                target_position.pose.pose.position.y = (bottom_right_x + bottom_right_y) / 2
                self.detect_pub.publish(target_position)
                
                if self.img_show:
                    cv2.rectangle(drawing, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)
                    cv2.putText(drawing, "{0} {1:.2f}".format(cls[cl], score), (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if self.img_show:
            cv2.imshow("drawing", drawing)
            cv2.waitKey(100)


yolo = Yolo()

if __name__ == "__main__":
    rospy.init_node("yolo_rknn_pub")
    rospy.Subscriber("/webcam_img", Image, yolo.Run)

    rospy.spin()

