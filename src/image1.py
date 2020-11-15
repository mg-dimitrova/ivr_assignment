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
    flag = 0 # detect if the object is visible or not
    cx, cy = 0.0 , 0.0
    #cnt = contours[0]
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
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

  PIXEL2METER = 0.039

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
    #initiate the joint publishers to send the sinusoidal joints angles to the robot
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize the time variables
    self.time_joint2 = rospy.get_time()
    #self.time_joint4 = rospy.get_time()
    #initiate publishers for joints' angular position estimated from camera 1
    #this could also be written into a single node joints as in the labs, but the handleing of the error (joints not visible)
    #would be less elegant
    #self.joint1_cam1_pub = rospy.Publisher("/joint1_camera1",Float64, queue_size=10)
    self.joint2_cam1_pub = rospy.Publisher("/joint2_camera1",Float64, queue_size=10)
    self.joint3_cam1_pub = rospy.Publisher("/joint3_camera1",Float64, queue_size=10)
    self.joint4_cam1_pub = rospy.Publisher("/joint4_camera1",Float64, queue_size=10)
    #publishers for the target coordinates
    self.sphere_target_y_pub = rospy.Publisher("/sphere_y_camera1",Float64, queue_size=10)
    self.sphere_target_z_pub = rospy.Publisher("/sphere_z_camera1",Float64, queue_size=10)

  def position_joint2(self, current_time):
    #joint 2 rotates around the x axis, the sinusoidal movement would be visible from camera 1 on the yz plane
    j2 = float((np.pi/2) * np.sin(np.pi/15 * current_time))
    return j2

  def position_joint3(self, current_time):
    #joint 3 rotates around the y axis, the angle would be visible from camera 2 on the xz plane
    #the motion can still be initiated from this script to be sure that when all the angles are changing the detection of joint 2 and 4 is correct
    j3 = float((np.pi/2) * np.sin(np.pi/18 * current_time))
    return j3

  def position_joint4(self, current_time):
    #joint 4 rotates around the x axis, the sinusoidal movement would be visible from camera 1 on the yz plane
    j4 = float((np.pi/2) * np.sin(np.pi/20 * current_time))
    return j4

  def detect_joint_angles(self, image):
    _, center = detect_colour(image, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    _, circle1Pos = detect_colour(image, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    _, circle2Pos = detect_colour(image, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    _, circle3Pos = detect_colour(image, self.RED_LOWER, self.RED_UPPER) #End effector

    #Getting divide by zero exception errors in some instances
    #Need to refactor this
    joint_angle_1 = np.arctan2((center[0] - circle1Pos[0]), (center[1] - circle1Pos[1])) #we assume joint
    joint_angle_2 = np.arctan2((circle1Pos[0] - circle2Pos[0]), (circle1Pos[1] - circle2Pos[1])) - joint_angle_1
    joint_angle_3 = np.arctan2((circle2Pos[0] - circle3Pos[0]), (circle2Pos[1] - circle3Pos[1])) - joint_angle_1 - joint_angle_2
    
    return np.array([joint_angle_1, joint_angle_2, joint_angle_3])

  def detect_individual_joint_angles(self, image, assume_zero=False, previous_state=False, predict=False):
    #flag_center, center = detect_colour(image, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    flag1Pos, circle1Pos = detect_colour(image, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    flag2Pos, circle2Pos = detect_colour(image, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    flag3Pos, circle3Pos = detect_colour(image, self.RED_LOWER, self.RED_UPPER) #End effector
  #joint 1 is assumed not to be changing for task 2.1, thus joint 3 can be detected only from camera2
    self.joint2_cam1 = Float64()
    self.previous_joint2 = Float64()
    self.joint3_cam1 = Float64()
    self.previous_joint3 = Float64()
    self.joint4_cam1 = Float64()
    self.previous_joint4 = Float64()
    #we assume joint1 is not rotating for task 2.1
    #  self.joint1_cam1 = np.arctan2((center[0] - circle1Pos[0]), (center[1] - circle1Pos[1]))
    if flag1Pos == 1 or flag2Pos == 1:
      #an error has occurred one of the joints is not visible, assign previous valid position to the current joint
      if assume_zero:
        self.joint2_cam1 = 0.0
        self.joint3_cam1 = 0.0
      elif previous_state:
        self.joint2_cam1 = self.previous_joint2
        self.joint3_cam1 = self.previous_joint3
      elif predict:
        pass
    else:
      self.joint2_cam1 = np.arctan2((circle1Pos[0] - circle2Pos[0]), (circle1Pos[1] - circle2Pos[1]))# - float(self.joint1_cam1)
      self.previous_joint2 = self.joint2_cam1
      dist = np.sqrt(np.sum((circle1Pos - circle2Pos)**2))
      print(dist)
      print(circle2Pos)
      self.joint3_cam1 = np.arccos(dist/90.0)
      self.previous_joint3 = self.joint3_cam1

    if flag2Pos == 1 or flag3Pos == 1:
      if assume_zero:
        self.joint4_cam1 = 0.0
      elif previous_state:
        self.joint4_cam1 = self.previous_joint4
    else:
      self.joint4_cam1 = np.arctan2((circle2Pos[0] - circle3Pos[0]),(circle2Pos[1] - circle3Pos[1])) - self.joint2_cam1 #- float(self.joint1_cam1)
      self.previous_joint4 = self.joint4_cam1

  def detect_targets(self, image, cube=False, sphere=True):
    self.cube_coords = Float64MultiArray()
    self.sphere_coords = Float64MultiArray()
    contours = detect_colour2(image, self.ORANGE_LOWER, self.ORANGE_UPPER)
    if len(contours) != 0:
      for contour in contours:
        if cube and is_cube(contour):
          M = cv2.moments(contour)
          if M['m00'] != 0:
            self.cube_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
        elif sphere and is_sphere:
          M = cv2.moments(contour)
          if M['m00'] != 0:
            self.sphere_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])

  def robot_clock_tick(self):
    #send control commands to joints for task 2.1
    curr_time = np.array([rospy.get_time() - self.time_joint2])
    self.joint2 = Float64()
    self.joint2.data = self.position_joint2(curr_time)
    self.joint3 = Float64()
    self.joint3.data = self.position_joint3(curr_time)
    self.joint4 = Float64()
    self.joint4.data = self.position_joint4(curr_time)

  def pixel2meter(self, image):
    #this value is always the same... we should not calculate it all the time... =
    # dist = 0.03845698760800996 m/px dist2 = 0.03888648856244221 m/px
    #link1= 64 px link2 = 90 px
    #we calculate the length of the link from link 1 as it is the only one whose joint angle does not change
    self.p2m = Float64()
    self.p2m2 = Float64()
    self.previous_p2m = Float64()
    flag_center, center = detect_colour(image, self.YELLOW_LOWER, self.YELLOW_UPPER)
    flag1Pos, circle1Pos = detect_colour(image, self.BLUE_LOWER, self.BLUE_UPPER)
    flag2Pos, circle2Pos = detect_colour(image, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    dist = np.sum((center - circle1Pos)**2)
    dist2 = np.sum((circle1Pos - circle2Pos)**2)
    print(np.sqrt(dist), np.sqrt(dist2))
    if flag1Pos != 1 and flag_center != 1:
      self.p2m = 2.5/np.sqrt(dist)
      self.previous_p2m = self.pixel2meter
      self.p2m2 = (3.5/np.sqrt(dist2))
    else:
      self.p2m = self.previous_p2m

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
    #estimate pixel to meter
    #self.pixel2meter(self.cv_image1)
    #print(self.p2m, self.p2m2)
    #estimate the joints from camera
    #This detects all angles at once, but, it cannot handle single joint errors (all fail or none)
    #self.joints = Float64MultiArray()
    #self.joints.data = self.detect_joint_angles(self.cv_image1)#This should be written out to a plot, also, this doesnt account for angles?

    #estimate joint angles individually
    self.detect_individual_joint_angles(self.cv_image1, assume_zero=True, previous_state=False, predict=False)
    #estimate the joints angle desired from sinusoidal formulas
    self.robot_clock_tick()
    #detect coordinate of the spherical target
    self.detect_targets(self.cv_image1)

    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      #publish joint position according to sinusoidal trend
      #self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      #self.robot_joint4_pub.publish(self.joint4)
      #publish the joint position calculated using vision
      self.joint2_cam1_pub.publish(self.joint2_cam1)
      self.joint3_cam1_pub.publish(self.joint3_cam1)
      self.joint4_cam1_pub.publish(self.joint4_cam1)
      #publish the target position calculated using vision
      self.sphere_target_y_pub.publish(self.sphere_coords[0])
      self.sphere_target_z_pub.publish(self.sphere_coords[1])
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


