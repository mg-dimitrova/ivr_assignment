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


class image_converter:

    # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    #initiate the joint publishers
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize the time variables
    self.time_joint2 = rospy.get_time()
    self.time_joint4 = rospy.get_time()

  def position_joint2(self):
    # get current time
    curr_time = np.array([rospy.get_time() - self.time_joint2])
    #joint 2 rotates around the x axis
    #the sinusoidal movement would be visible from camera 1 on the yz plane
    j2 = float((np.pi/2) * np.sin(np.pi/15 * curr_time))
    return j2

  def position_joint4(self):
    # get current time
    curr_time = np.array([rospy.get_time() - self.time_joint4])
    #joint 4 rotates around the x axis
    #the sinusoidal movement would be visible from camera 1 on the yz plane
    j4 = float((np.pi/2) * np.sin(np.pi/20 * curr_time))
    return j4

  def detect_yellow(self, image):
    #converting image from BGR to HSV color-space (easier to segment an image based on its color)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #define boundaries for yellow in HSV
    #the boundaries are found using the method described in https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html?highlight=inrange
    lower = np.array([20,100,100])
    upper = np.array([40,255,255])
    #generating mask to get yellow joint
    mask = cv2.inRange(hsv, lower, upper)
    #generate kernel for morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    #applying closing (dilation followed by erosion)
    #dilation allows to close black spots inside the mask
    #erosion allows to return to dimension close to the original ones for more accurate estimation of the center
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #estimating the treshold and contour for calculating the moments (as in https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=moments)
    ret, thresh = cv2.threshold(closing, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    #estimate moments
    M = cv2.moments(cnt)
    #find the centre of mass from the moments estimation
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx, cy])

  def detect_blue(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([110,50,50])
    upper = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(closing, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx, cy])

  def detect_green(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([50, 100, 100])
    upper = np.array([70,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(closing, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx, cy])

  def detect_red(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0,100,100])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(closing, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx, cy])

  def detect_orange(self, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([9,100,100])
    upper = np.array([29, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imwrite('mask_orange.png', mask)
    '''This part has to be changed to recognize whether it is a circle or not'''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(closing, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx, cy])

  def detect_joint_angles(self, image):
    center = self.detect_yellow(image)
    circle1Pos = self.detect_blue(image)
    circle2Pos = self.detect_green(image)
    circle3Pos = self.detect_red(image)
    ja1 = np.arctan((center[0]-circle1Pos[0])/(center[1]-circle1Pos[1]))
    ja2 = np.arctan((circle1Pos[0]-circle2Pos[0])/(circle1Pos[1]-circle2Pos[1]))-ja1
    ja3 = np.arctan((circle2Pos[0]-circle3Pos[0])/(circle2Pos[1]-circle3Pos[1]))-ja1-ja2
    return np.array([ja1, ja2, ja3])

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
  # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy1.png', self.cv_image1)

    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)

    #estimate the joints from camera
    #self.joints = Float64MultiArray()
    #self.joints.data = self.detect_joint_angles(self.cv_image1)

    #psend control commands to joints for task 2.1
    #self.joint2 = Float64()
    #self.joint2.data = self.position_joint2()
    #self.joint4 = Float64()
    #self.joint4.data = self.position_joint4()

    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      #self.robot_joint2_pub.publish(self.joint2)
      #self.robot_joint4_pub.publish(self.joint4)
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


