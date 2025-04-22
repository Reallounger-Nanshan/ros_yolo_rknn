#!/home/ubuntu/anaconda3/envs/toolkit2/bin/python3.10

import cv2
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

def TargetPositionCallback(odometry_msg):
    print("Target Position:\n    x: {:.4f}\n    y: {:.4f}\n".format(odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y))

if __name__=="__main__":
    capture = cv2.VideoCapture(0)
    
    rospy.init_node('camera_node', anonymous=True)
    rospy.Subscriber("/ros_yolo_rknn/target_position", Odometry, TargetPositionCallback)
    image_pub = rospy.Publisher('/webcam_img', Image, queue_size = 1)

    while not rospy.is_shutdown():
        ret, frame = capture.read()
        if ret:
            ros_frame = Image()
            header = Header(stamp = rospy.Time.now())
            header.frame_id = "Camera"
            ros_frame.header = header
            ros_frame.width = 640
            ros_frame.height = 480
            ros_frame.encoding = "bgr8"
            ros_frame.step = 1920
            # ros_frame.data = np.array(frame).tostring()
            ros_frame.data = np.array(frame).tobytes()
            image_pub.publish(ros_frame)
            # cv2.imshow('camera', frame)
            # cv2.waitKey(3)

    capture.release()
    cv2.destroyAllWindows() 
    print("quit successfully!")

    rospy.spin()
