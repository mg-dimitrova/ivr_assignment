#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

def detect_colour(image, lower_colour_boundary, upper_colour_boundary):
    #DOESN'T SOLVE FOR OVERLAPPING COLOURS

    #converting image from BGR to HSV color-space (easier to segment an image based on its color)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_colour_boundary, upper_colour_boundary)
    #generate kernel for morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #applying closing (dilation followed by erosion)
    #dilation allows to close black spots inside the mask
    #erosion allows to return to dimension close to the original ones for more accurate estimation of the center
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #estimating the treshold and contour for calculating the moments (as in https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=moments)
    ret, thresh = cv2.threshold(closing, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2) #This returns multiple contours, so for orange we expect more than one
    cnt = contours[0]
    #estimate moments
    M = cv2.moments(cnt)
    #find the centre of mass from the moments estimation
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx, cy])  
    
class image_converter:

  YELLOW_LOWER = np.array([20,100,100])
  YELLOW_UPPER = np.array([40,255,255])

  BLUE_LOWER = np.array([110,50,50])
  BLUE_UPPER = np.array([130,255,255])

  GREEN_LOWER = np.array([50, 100, 100])
  GREEN_UPPER = np.array([70,255,255])

  RED_LOWER = np.array([0,100,100])
  RED_UPPER = np.array([10, 255, 255])  

  ORANGE_LOWER = np.array([9,100,100])
  ORANGE_UPPER = np.array([29, 255, 255])

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    #initiate the joint publishers
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    # initialize the variable for the starting time
    self.time_joint3 = rospy.get_time()

  #define sinusoidal trajectory for task 2.1
  def position_joint3(self):
    # get current time
    curr_time = np.array([rospy.get_time() - self.time_joint3])
    #joint 3 rotates around the y axis, the movement would be visible from camera 2 in the zx plane
    j3 = float((np.pi/2) * np.sin(np.pi/18 * curr_time))
    return j3

  
  def detect_joint_angles(self, image):
    center = detect_colour(image, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    circle1Pos = detect_colour(image, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    circle2Pos = detect_colour(image, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    circle3Pos = detect_colour(image, self.RED_LOWER, self.RED_UPPER) #End effector

    #object_to_be_tracked = detect_colour(image, self.ORANGE_LOWER, self.ORANGE_UPPER)

    #Getting divide by zero exception errors in some instances
    #Need to refactor this
    joint_angle_1 = np.arctan((center[0] - circle1Pos[0]) 
                                / (center[1] - circle1Pos[1]))
    joint_angle_2 = np.arctan((circle1Pos[0] - circle2Pos[0]) 
                                / (circle1Pos[1] - circle2Pos[1])) - joint_angle_1
    joint_angle_3 = np.arctan((circle2Pos[0] - circle3Pos[0]) 
                                / (circle2Pos[1] - circle3Pos[1])) - joint_angle_1 - joint_angle_2
    
    return np.array([joint_angle_1, joint_angle_2, joint_angle_3])

  def robot_clock_tick(self):
    #psend control commands to joints for task 2.1
    curr_time = np.array([rospy.get_time() - self.time_joint2])
    
    self.joint3 = Float64()
    self.joint3.data = self.position_joint3(curr_time)

  # Recieve data, process it, and publish
  def callback2(self,data):
  # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy2.png', self.cv_image2)
    #im2=cv2.imshow('window2', self.cv_image2)
    #cv2.waitKey(1)

    #detect joints angles from camera
    self.joints = Float64MultiArray()
    self.joints.data = self.detect_joint_angles(self.cv_image2)

    #send control commands to joints for task 2.1
    self.joint3 = Float64()
    self.joint3.data = self.position_joint3()

    # Publish the results
    try:
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      self.robot_joint3_pub.publish(self.joint3)
    except CvBridgeError as e:
      print(e)

    # call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
  main(sys.argv)
