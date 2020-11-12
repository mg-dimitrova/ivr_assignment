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
    flag = 0
    cx, cy = 0.0, 0.0
    if len(contours) == 0:
      flag = 1
    else:
      cnt = contours[0]
      #estimate moments
      M = cv2.moments(cnt)
      if M['m00'] == 0:
        flag = 1
      else:
        #find the centre of mass from the moments estimation
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return flag, np.array([cx, cy])

def detect_colour2(image, lower_colour_boundary, upper_colour_boundary):
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
    return contours

def is_cube(contour):
  approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
  area = cv2.contourArea(contour)
  if len(approx) < 8:
    return True
  return False

def is_sphere(contour):
  approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
  area = cv2.contourArea(contour)
  print(len(approx))
  if ((len(approx) > 8) & (area > 30)):
    return True
  return False

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
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize the variable for the starting time
    self.time_previous_step = rospy.get_time()
    self.time_joint3 = rospy.get_time()
    self.joint1_cam2_pub = rospy.Publisher("/joint1_camera2",Float64, queue_size=10)
    self.joint2_cam2_pub = rospy.Publisher("/joint2_camera2",Float64, queue_size=10)
    self.joint3_cam2_pub = rospy.Publisher("/joint3_camera2",Float64, queue_size=10)
    self.joint4_cam2_pub = rospy.Publisher("/joint4_camera2",Float64, queue_size=10)

  #define sinusoidal trajectory for task 2.1
  def position_joint2(self, current_time):
    # get current time
    #curr_time = np.array([rospy.get_time() - self.time_joint2])
    #joint 2 rotates around the x axis
    #the sinusoidal movement would be visible from camera 1 on the yz plane
    j2 = float((np.pi/2) * np.sin(np.pi/15 * current_time))
    return j2

  def position_joint3(self, current_time):
    j3 = float((np.pi/2) * np.sin(np.pi/18 * current_time))
    return j3

  def position_joint4(self, current_time):
    # get current time
    #curr_time = np.array([rospy.get_time() - self.time_joint4])
    #joint 4 rotates around the x axis
    #the sinusoidal movement would be visible from camera 1 on the yz plane
    j4 = float((np.pi/2) * np.sin(np.pi/20 * current_time))
    return j4
  def detect_joint_angles(self, image):
    center = detect_colour(image, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    circle1Pos = detect_colour(image, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    circle2Pos = detect_colour(image, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    circle3Pos = detect_colour(image, self.RED_LOWER, self.RED_UPPER) #End effector

    #object_to_be_tracked = detect_colour(image, self.ORANGE_LOWER, self.ORANGE_UPPER)

    #Getting divide by zero exception errors in some instances
    #Need to refactor this
    joint_angle_1 = np.arctan2((center[0] - circle1Pos[0]),(center[1] - circle1Pos[1]))
    joint_angle_2 = np.arctan2((circle1Pos[0] - circle2Pos[0]), (circle1Pos[1] - circle2Pos[1])) - joint_angle_1
    joint_angle_3 = np.arctan2((circle2Pos[0] - circle3Pos[0]), (circle2Pos[1] - circle3Pos[1])) - joint_angle_1 - joint_angle_2
    
    return np.array([joint_angle_1, joint_angle_2, joint_angle_3])

  def detect_individual_joint_angles(self, image):
    flag_center, center = detect_colour(image, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    flag1Pos, circle1Pos = detect_colour(image, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    flag2Pos, circle2Pos = detect_colour(image, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    flag3Pos, circle3Pos = detect_colour(image, self.RED_LOWER, self.RED_UPPER) #End effector
    self.joint1_cam2 = Float64() #joint 1 is assumed not to be changing for task 2.1
    #self.joint2_cam2 = Float64()
    self.joint3_cam2 = Float64() #if joint 1 does not rotate, joint 3 can be detected only from camera2
    self.joint4_cam2 = Float64()
    #Getting divide by zero exception errors in some instances
    #Need to refactor this
    if flag_center == 1 or flag1Pos == 1:
      #an error has occurred, one of the joints is not visible
      pass
    else:
      self.joint1_cam2 = np.arctan2((center[0] - circle1Pos[0]), (center[1] - circle1Pos[1])) #we assume joint1 is not rotating for task 2.1
    if flag1Pos == 1 or flag2Pos == 1:
      #an error has occurred one of the joints is not visible
      pass
    else:
      self.joint3_cam2 = np.arctan2((circle1Pos[0] - circle2Pos[0]), (circle1Pos[1] - circle2Pos[1])) - self.joint1_cam2
    #if joint 1 does not rotate, joint 2 can be detected only from camera 1
    if flag2Pos == 1 or flag3Pos == 1:
      pass
    else:
      self.joint4_cam2 = np.arctan2((circle2Pos[0] - circle3Pos[0]), (circle2Pos[1] - circle3Pos[1])) - self.joint1_cam2 - self.joint3_cam2

  def detect_targets(self, image):
    contours = detect_colour2(image, self.ORANGE_LOWER, self.ORANGE_UPPER)
    #Finds the circle orange object
    for contour in contours:
      if is_cube(contour):
        print('Cube found at:')
        M = cv2.moments(contour)
        cube_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
        print(cube_coords)
      elif is_sphere: #SPHERE IS WHAT IS IMPORTANT FOR THE ASSIGNMENT
        #Calculate sphere distance from base object
        print('Sphere found at:')
        M = cv2.moments(contour)
        sphere_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
        print(sphere_coords)
    return cube_coords, sphere_coords

  def robot_clock_tick(self):
    #send control commands to joints for task 2.1
    curr_time = np.array([rospy.get_time() - self.time_joint3])
    self.joint2 = Float64()
    self.joint2.data = self.position_joint2(curr_time)
    self.joint3 = Float64()
    self.joint3.data = self.position_joint3(curr_time)
    self.joint4 = Float64()
    self.joint4.data = self.position_joint4(curr_time)

  # Recieve data, process it, and publish
  def callback2(self,data):
  # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy2.png', self.cv_image2)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    #detect joints angles from camera
    #self.joints = Float64MultiArray()
    #self.joints.data = self.detect_joint_angles(self.cv_image2)
    self.detect_individual_joint_angles(self.cv_image2)
    #send sinusoindal value to joints
    self.robot_clock_tick()

    # Publish the results
    try:
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      self.robot_joint4_pub.publish(self.joint4)
      self.joint1_cam2_pub.publish(self.joint1_cam2)
      #self.joint2_cam2_pub.publish(self.joint2_cam2)
      self.joint3_cam2_pub.publish(self.joint3_cam2)
      self.joint4_cam2_pub.publish(self.joint4_cam2)
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
