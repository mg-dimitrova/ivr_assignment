#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import math
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

def detect_colour(image, lower_colour_boundary, upper_colour_boundary, is_target = False):
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
  if is_target:
    return contours
  else:
    flag = 0 # detect if the object is visible or not
    cx, cy = 0.0 , 0.0
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

def is_cube(contour):
  approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
  area = cv2.contourArea(contour)
  if len(approx) < 8:
    return True
  return False

def is_sphere(contour):
  approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
  area = cv2.contourArea(contour)
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

  WHITE_LOWER = np.array([0,0, 245])
  WHITE_UPPER = np.array([255,255,255])

  PIXEL2METER = 0.039

  PIXEL_PER_METER = 25.8

    # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to receive messages from a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize a subscriber to receive messages from a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize a subscriber to receive messages from a topic named /robot/joint_states
    self.joint_states_sub = rospy.Subscriber("/robot/joint_states",JointState,self.joint_states_callback)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    #initiate the joint publishers to send the sinusoidal joints angles to the robot
    self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize the time variables
    self.time_joints = rospy.get_time()
    #initiate publishers for joints' angular position estimated from camera 1 and 2
    self.joint1_vision_pub = rospy.Publisher("/joint1",Float64, queue_size=10)
    self.joint2_vision_pub = rospy.Publisher("/joint2",Float64, queue_size=10)
    self.joint3_vision_pub = rospy.Publisher("/joint3",Float64, queue_size=10)
    self.joint4_vision_pub = rospy.Publisher("/joint4",Float64, queue_size=10)
    self.previous_joints = np.array([0.0, 0.0, 0.0, 0.0], dtype ='float64')
    self.error_joint_pub = rospy.Publisher("/error_joint", Float64, queue_size=10)
    self.error_joint = 0.0
    #publishers for the target coordinates
    self.sphere_target_x_pub = rospy.Publisher("/sphere_x", Float64, queue_size=10)
    self.sphere_target_y_pub = rospy.Publisher("/sphere_y", Float64, queue_size=10)
    self.sphere_target_z_pub = rospy.Publisher("/sphere_z", Float64, queue_size=10)
    #publishers for the end_effector
    self.xe_pub = rospy.Publisher("/xe",Float64, queue_size=10)
    self.ye_pub = rospy.Publisher("/ye",Float64, queue_size=10)
    self.ze_pub = rospy.Publisher("/ze",Float64, queue_size=10)
    #initialize error
    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
    self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
    self.error = np.array([0.0,0.0,0.0], dtype ='float64')
    self.error_d = np.array([0.0,0.0,0.0], dtype ='float64')
    self.initialize_joints()
    self.time_offset = None
    self.time_offset = -1

  def position_joint2(self, current_time):
    j2 = float((np.pi/2) * np.sin(np.pi/15 * current_time))
    return j2

  def position_joint3(self, current_time):
    j3 = float((np.pi/2) * np.sin(np.pi/18 * current_time))
    return j3

  def position_joint4(self, current_time):
    j4 = float((np.pi/2) * np.sin(np.pi/20 * current_time))
    return j4

  def detect_joint_angle_2_blue(self, image1, image2):
    _, circleBlue = detect_colour(image1, self.BLUE_LOWER, self.BLUE_UPPER)
    _, circleGreen = detect_colour(image1,self.GREEN_LOWER, self.GREEN_UPPER)
    #theta = - np.arcsin((circleGreen[0] - circleBlue[0])/ 100) #Max length for link
    #theta = -np.sign(circleGreen[0]- circleBlue[0])*np.arccos((circleBlue[1] - circleGreen[1])/100) # -> better for individual than previous, not too good for all joints together
    theta = np.arctan2((circleBlue[0] - circleGreen[0]), (circleBlue[1] - circleGreen[1])) #most precise
    return theta

  def detect_joint_angle_3_blue(self, image1, image2):
    _, circleBlue = detect_colour(image2, self.BLUE_LOWER, self.BLUE_UPPER)
    _, circleGreen = detect_colour(image2,self.GREEN_LOWER, self.GREEN_UPPER)
    theta = np.arcsin((circleGreen[0] - circleBlue[0])/ 100) #Max length for link
    #theta = -np.arctan2((circleBlue[0] - circleGreen[0]), (circleBlue[1] - circleGreen[1]))
    return theta

  def detect_joint_angle_4_green(self, image1, image2):
    _, Blue1 = detect_colour(image1, self.BLUE_LOWER, self.BLUE_UPPER)
    _, Green1 = detect_colour(image1,self.GREEN_LOWER, self.GREEN_UPPER)
    _, Red1 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER)
    _, Blue2 = detect_colour(image2, self.BLUE_LOWER, self.BLUE_UPPER)
    _, Green2 = detect_colour(image2,self.GREEN_LOWER, self.GREEN_UPPER)
    _, Red2 = detect_colour(image2, self.RED_LOWER, self.RED_UPPER)
    circleBlue = np.array([Blue2[0], Blue1[0], Blue1[1]])
    circleGreen = np.array([Green2[0], Green1[0], Green1[1]])
    circleRed = np.array([Red2[0], Red1[0], Red1[1]])

    #Blue
    a = circleBlue
    #Green
    b = circleGreen
    #Red
    c = circleRed

    ba = a - b
    bc = b - c
    x = circleGreen[0]-circleRed[0]
    y = circleGreen[1]-circleRed[1]
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.sign(y)*np.arccos(cosine_angle)
    #angle = np.arctan2((Green1[0] - Red1[0]),(Green1[1] - Red1[1]))
    #angle = -np.arctan2((Green2[0] - Red2[0]),(Green2[1] - Red2[1]))
    #return np.degrees(angle)
    return angle

  def detect_colour2(self, image1, image2, lower_colour_boundary, upper_colour_boundary, target=None):
    #same as detect_colour but returns coordinates in 3D
    #converting image from BGR to HSV color-space (easier to segment an image based on its color)
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_colour_boundary, upper_colour_boundary)

    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_colour_boundary, upper_colour_boundary)

    #generate kernel for morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    #applying closing (dilation followed by erosion)
    #dilation allows to close black spots inside the mask
    #erosion allows to return to dimension close to the original ones for more accurate estimation of the center
    closing1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    closing2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

    #estimating the treshold and contour for calculating the moments (as in https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=moments)
    ret1, thresh1 = cv2.threshold(closing1, 127, 255, 0)
    ret2, thresh2 = cv2.threshold(closing2, 127, 255, 0)

    contours1, hierarchy1 = cv2.findContours(thresh1, 1, 2) #This returns multiple contours, so for orange we expect more than one

    if target != None:
      for contour in contours1:
        if is_sphere(contour):
          cnt1 = contour
    else:
      cnt1 = contours1[0]

    M1 = cv2.moments(cnt1)

    contours2, hierarchy2 = cv2.findContours(thresh2, 1, 2) #This returns multiple contours, so for orange we expect more than one

    if target != None:
      for contour in contours2:
        if is_sphere(contour):
          cnt2 = contour
    else:
      cnt2 = contours2[0]
    M2 = cv2.moments(cnt2)

    #cx, cy, cz = 0.0 , 0.0, 0.0
    #find the centre of mass from the moments estimation
    #Cam2 = X
    cx = int(M2['m10']/M2['m00'])

    #Cam1 = Z/Y
    cy = int(M1['m10']/M1['m00'])
    cz = int(M1['m01']/M1['m00'])
    return np.array([cx, cy, cz])

  def get_target_x(self, source_point, target_point):
    return self.pixels_to_meters(target_point[0] - source_point[0])

  def get_target_y(self, source_point, target_point):
    return self.pixels_to_meters(target_point[1] - source_point[1])

  def get_target_z(self, source_point, target_point):
    return self.pixels_to_meters(source_point[2] - target_point[2] + 0.8) #Base Offset

  def detect_target_range(self, source_point, target_point):
    return np.array([self.get_target_x(source_point, target_point), self.get_target_y(source_point, target_point),self.get_target_z(source_point, target_point)])

  def move_joints_2_1(self):
    ######################################################################
    #send control commands to joints for task 2.1
    ######################################################################
    curr_time = np.array([rospy.get_time() - self.time_joints])
    self.joint2 = Float64()
    self.joint2.data = self.position_joint2(curr_time)
    self.joint3 = Float64()
    self.joint3.data = self.position_joint3(curr_time)
    self.joint4 = Float64()
    self.joint4.data = self.position_joint4(curr_time)
    ######################################################################
    ######################################################################
    ######################################################################

  def get_joint_state_2_1(self):
    ######################################################################
    #Joint state estimation for task 2.1
    ######################################################################
    joint_1 = 0.0
    joint_2 = self.detect_joint_angle_2_blue(self.cv_image1, self.cv_image2)
    joint_3 = self.detect_joint_angle_3_blue(self.cv_image1, self.cv_image2)
    joint_4 = self.detect_joint_angle_4_green(self.cv_image1, self.cv_image2)
    joints = np.array([joint_1, joint_2, joint_3, joint_4])
    return joints
    ######################################################################
    ######################################################################
    ######################################################################

  def publish_joints_vision(self,joints):
    self.joint1_vision = Float64()
    self.joint2_vision = Float64()
    self.joint3_vision = Float64()
    self.joint4_vision = Float64()
    self.joint1_vision.data = joints[0]
    self.joint2_vision.data = joints[1]
    self.joint3_vision.data = joints[2]
    self.joint4_vision.data = joints[3]

  def detect_targets_2_2(self, from_base=False):
    ######################################################################
    #Detect the sphere target for task 2.2
    ######################################################################
    if not from_base:
      coordinates = self.detect_target_range(self.detect_colour2(self.cv_image1, self.cv_image2, self.YELLOW_LOWER, self.YELLOW_UPPER), self.detect_colour2(self.cv_image1, self.cv_image2, self.ORANGE_LOWER, self.ORANGE_UPPER, "Sphere"))
    else:
      coordinates = self.detect_target_range(self.detect_colour2(self.cv_image1, self.cv_image2, self.WHITE_LOWER, self.WHITE_UPPER), self.detect_colour2(self.cv_image1, self.cv_image2, self.ORANGE_LOWER, self.ORANGE_UPPER, "Sphere"))
    self.sphere_x = Float64()
    self.sphere_y = Float64()
    self.sphere_z = Float64()
    self.sphere_x.data = coordinates[0]
    self.sphere_y.data = coordinates[1]
    self.sphere_z.data = coordinates[2]
    return coordinates
    ######################################################################
    ######################################################################
    ######################################################################

  def detect_end_effector(self, image1, image2):
    yellow = np.array([398, 532])
    #there is a little offset on z, but the x and y coordinates calculated by the cameras are the same +- 1pixel
    flagRed1, circleRed1 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER) #End effector
    flagRed2, circleRed2 = detect_colour(image2, self.RED_LOWER, self.RED_UPPER)
    r_yz = self.PIXEL2METER*(yellow-circleRed1)
    r_xz = self.PIXEL2METER*(yellow-circleRed2)
    return np.array([r_xz[0], r_yz[0], r_yz[1]])

  def pixels_to_meters(self, pixels):
      return pixels / self.PIXEL_PER_METER

  def forward_kinematics(self, joints = None):
    if joints == None:
      [t1, t2, t3, t4] = self.joints_ros
    else:
      t1, t2, t3, t4 = joints[0], joints[1], joints[2], joints[3] #theta1 to theta4
    print(t1, t2, t3, t4)
    l1 = 2.5
    l3 = 3.5
    l4 = 3.0
    xe = l4*np.cos(t4)*(np.sin(t1)*np.sin(t2)*np.cos(t3) + np.cos(t1)*np.sin(t3)) + l4*np.sin(t4)*np.sin(t1)*np.cos(t2) + l3*np.cos(t3)*np.sin(t1)*np.sin(t2) + l3*np.cos(t1)*np.sin(t3)
    ye = l4*np.cos(t4)*(-np.cos(t1)*np.sin(t2)*np.cos(t3) + np.sin(t1)*np.sin(t3)) - l4*np.sin(t4)*np.cos(t1)*np.cos(t2) - l3*np.cos(t3)*np.cos(t1)*np.sin(t2) + l3*np.sin(t1)*np.sin(t3)
    ze = l4*np.cos(t4)*(np.cos(t2)*np.cos(t3)) - l4*np.sin(t4)*np.sin(t2) + l3*np.cos(t3)*np.cos(t2) + l1
    #print(xe, ye, ze)
    return np.array([xe, ye, ze])

  def calculate_jacobian(self, joints):
    t1, t2, t3, t4 = joints[0], joints[1], joints[2], joints[3]
    #l1 = 2.5 #l2=0
    l3 = 3.5
    l4 = 3.0
    J11 = l4*np.cos(t4)*(np.cos(t1)*np.sin(t2)*np.cos(t3) - np.sin(t1)*np.sin(t3)) + l4*np.sin(t4)*np.cos(t1)*np.cos(t2) + l3*np.cos(t3)*np.cos(t1)*np.sin(t2) - l3*np.sin(t1)*np.sin(t3)
    J21 = l4*np.cos(t4)*(np.sin(t1)*np.sin(t2)*np.cos(t3) + np.cos(t1)*np.sin(t3)) + l4*np.sin(t4)*np.sin(t1)*np.cos(t2) + l3*np.cos(t3)*np.sin(t1)*np.sin(t2) + l3*np.cos(t1)*np.sin(t3)
    J31 = 0.0
    J12 = l4*np.cos(t4)*(np.sin(t1)*np.cos(t2)*np.cos(t3)) - l4*np.sin(t4)*np.sin(t1)*np.sin(t2) + l3*np.cos(t3)*np.sin(t1)*np.cos(t2)
    J22 = -l4*np.cos(t4)*np.cos(t1)*np.cos(t2)*np.cos(t3) + l4*np.sin(t4)*np.cos(t1)*np.sin(t2) - l3*np.cos(t3)*np.cos(t1)*np.cos(t2)
    J32 = -l4*np.cos(t4)*np.sin(t2)*np.cos(t3) - l4*np.sin(t4)*np.cos(t2) - l3*np.cos(t3)*np.sin(t2)
    J13 = l4*np.cos(t4)*(- np.sin(t1)*np.sin(t2)*np.sin(t3) + np.cos(t1)*np.cos(t3)) - l3*np.sin(t3)*np.sin(t1)*np.sin(t2) + l3*np.cos(t1)*np.cos(t3)
    J23 = l4*np.cos(t4)*(np.cos(t1)*np.sin(t2)*np.sin(t3) + np.sin(t1)*np.cos(t3)) + l3*np.sin(t3)*np.cos(t1)*np.sin(t2) + l3*np.sin(t1)*np.cos(t3)
    J33 = -l4*np.cos(t4)*np.cos(t2)*np.sin(t3) - l3*np.sin(t3)*np.cos(t2)
    J14 = -l4*np.sin(t4)*(np.sin(t1)*np.sin(t2)*np.cos(t3) + np.cos(t1)*np.sin(t3)) + l4*np.cos(t4)*np.sin(t1)*np.cos(t2)
    J24 = -l4*np.sin(t4)*(-np.cos(t1)*np.sin(t2)*np.cos(t3) + np.sin(t1)*np.sin(t3)) - l4*np.cos(t4)*np.cos(t1)*np.cos(t2)
    J34 = -l4*np.sin(t4)*np.cos(t2)*np.cos(t3) - l4*np.cos(t4)*np.sin(t2)
    J = np.array([[J11, J12, J13, J14],[J21, J22, J23, J24], [J31, J32, J33, J34]])
    return J

  def calculate_pseudo_invert_jacobian(self,joints):
    J = self.calculate_jacobian(joints)
    pinv_J = np.linalg.pinv(J)
    return pinv_J

  def publish_end_effector(self, pe):
    self.xe = Float64()
    self.ye = Float64()
    self.ze = Float64()
    self.xe.data = pe[0]
    self.ye.data = pe[1]
    self.ze.data = pe[2]

  def closed_loop(self):
    joints = self.joints_ros
    Kp = 2.0 * np.eye(3)
    Ki = 0.2 * np.eye(3)
    Kd = 0.15 * np.eye(3)
    invJ = self.calculate_pseudo_invert_jacobian(joints)
    current_time = rospy.get_time()
    dt = current_time - self.time_previous_step2
    self.time_previous_step2 = current_time
    x_d = self.detect_targets_2_2()
    print(x_d)
    x_e = self.detect_end_effector(self.cv_image1, self.cv_image2)
    self.publish_end_effector(x_e)
    print(x_e)
    err = x_d - x_e
    self.error_d = (err - self.error)/dt
    self.error_i = (err - self.error)*dt
    self.error = err
    qdot = np.dot(invJ, (np.dot(Kp, self.error.transpose()) + np.dot(Kd, self.error_d.transpose()) + np.dot(Ki, self.error_i.transpose())))
    q = joints + dt * qdot

    return q

  def forward_kinematics_3_1(self):
    ######################################################################
    #Calculate FK for task 3.1
    ######################################################################
    #print(self.joints_ros)
    calculated_value = self.forward_kinematics([0.0, 0.25, 0.5, 0.4])
    actual_value = self.detect_colour2(self.cv_image1, self.cv_image2, self.RED_LOWER, self.RED_UPPER)
    print(calculated_value)
    print(self.detect_target_range(self.detect_colour2(self.cv_image1, self.cv_image2, self.YELLOW_LOWER, self.YELLOW_UPPER), actual_value))
    ######################################################################
    ######################################################################
    ######################################################################

  def closed_loop_control_3_2(self):
    ######################################################################
    #Closed loop control for task 3.2
    ######################################################################
    joints_closed = self.closed_loop()
    self.joint1 = Float64()
    self.joint1.data = joints_closed[0]
    self.joint2 = Float64()
    self.joint2.data = joints_closed[1]
    self.joint3 = Float64()
    self.joint3.data = joints_closed[2]
    self.joint4 = Float64()
    self.joint4.data = joints_closed[3]
    ######################################################################
    ######################################################################
    ######################################################################

  def calculateDistance(self, x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

  def initialize_joints(self):
    """
    0 = base
    1 = joint 1
    1 = joint 2
    2 = joint 3
    3 = end effector
    """
    self.previous_joints = [[399.5, 396.0, 536.0], [400.0, 393.0, 457.0], [434.0, 352.0, 381.0], [456.0, 309.0, 333.0]]
      
    self.time_offset = Float64()
    self.time_offset = rospy.get_time()
    
    curr_time = rospy.get_time() - self.time_offset + 1
    self.joint2 = Float64()
    self.joint2.data = self.position_joint2(curr_time)
    self.joint3 = Float64()
    self.joint3.data = self.position_joint3(curr_time)
    self.joint4 = Float64()
    self.joint4.data = self.position_joint4(curr_time)
    self.robot_joint2_pub.publish(self.joint2)
    self.robot_joint3_pub.publish(self.joint3)
    self.robot_joint4_pub.publish(self.joint4)

  def move_joints_4_3(self):
    if self.time_offset > -1:
      curr_time = rospy.get_time() - self.time_offset  + 1
      #curr_time = 1
      
      self.joint2 = Float64()
      self.joint2.data = self.position_joint2(curr_time)
      self.joint3 = Float64()
      self.joint3.data = self.position_joint3(curr_time)
      self.joint4 = Float64()
      self.joint4.data = self.position_joint4(curr_time)
      #self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      #self.robot_joint4_pub.publish(self.joint4)  
    else:
      self.initialize_joints()

  def detect_joint_angle_2_black(self, joint2, joint4):   
    #theta = np.arcsin((joint4[0] - joint2[0])/ 100) #Max length for link
    #print(joint2)
    #print(joint4)
    
    theta = np.arctan2((joint2[1] - joint4[1]), (joint2[2] - joint4[2]))
    #print("2: ", theta)
    return theta

  def detect_joint_angle_3_black(self, joint2, joint4):
    print(joint2)
    print(joint4)

    x = (joint4[0] - joint2[0])
    y = ((joint2[2] - joint4[2]))

    print(x,y)

    theta = np.arctan2(x, y)
    #theta = np.arctan2(30, 70)
    print("3: ",theta)
    return theta

  def detect_joint_angle_4_black(self, joint2, joint3, end_effector):
 
    a = np.array(joint2)
    b = np.array(joint3)
    c = np.array(end_effector)

    ba = a - b
    bc = b - c

    x = joint3[0]-end_effector[0]
    y = joint3[1]-end_effector[1]
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.sign(y)*np.arccos(cosine_angle)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    theta = np.arccos(cosine_angle)

    #print("4: ", theta)
    return theta
    
  def update_camera_1(self):
    joints1 = []
    hsv1 = cv2.cvtColor(self.cv_image1, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, np.array([0,0,0]), np.array([10,10,10])) #BLACK
    img1 = self.cv_image1
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    corners1 = cv2.goodFeaturesToTrack(mask1,5,0.20,20) #Val index 2 sets sensitivity of edge detection
    corners1 = np.int0(corners1)
    
    #Find points within 30 pixels of each other
    for i in corners1:
      i_x, i_y = i.ravel()
      for j in corners1:
        j_x, j_y = j.ravel() 
        dist = self.calculateDistance(i_x, i_y, j_x, j_y)
        if dist < 30:
          joints1.append([i,j])
    
    #Draw lines between those points
    for points in joints1:
      x_1, y_1 = points[0].ravel()
      x_2, y_2 = points[1].ravel()
      cv2.line(img1, (x_1, y_1), (x_2, y_2), (150, 0, 0), thickness=4, lineType=8)
    
    mask_joints1 = cv2.inRange(img1, np.array([50,0,0]), np.array([250,0,0]))
    
    #generate kernel for morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    
    closing_joints1 = cv2.morphologyEx(mask_joints1, cv2.MORPH_CLOSE, kernel)
    
    #estimating the treshold and contour for calculating the moments (as in https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=moments)
    ret1, thresh1 = cv2.threshold(closing_joints1, 127, 255, 0)

    contours_joints1, hierarchy1 = cv2.findContours(thresh1, 1, 2) 

    if len(contours_joints1) == 4:
      for contour in contours_joints1:
        ((new_y, new_z), radius) = cv2.minEnclosingCircle(contour)
        
        joint_number = None
        smallest_distance = 1000

        for num, joint in enumerate(self.previous_joints):
          joint_distance = self.calculateDistance(new_y, new_z, joint[1], joint[2])

          if joint_distance < smallest_distance:
            smallest_distance = joint_distance
            joint_number = num

        
        self.previous_joints[joint_number][1] = new_y
        self.previous_joints[joint_number][2] = new_z

        #print(joint_number, ",", smallest_distance)

  def update_camera_2(self):
    joints2 = []
    hsv2 = cv2.cvtColor(self.cv_image2, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, np.array([0,0,0]), np.array([10,10,10])) #BLACK
    img2 = self.cv_image2
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    corners2 = cv2.goodFeaturesToTrack(mask2,5,0.20,20) #Val index 2 sets sensitivity of edge detection
    corners2 = np.int0(corners2)

    #Find points within 30 pixels of each other
    for i in corners2:
      i_x, i_y = i.ravel()
      for j in corners2:
        j_x, j_y = j.ravel() 
        dist = self.calculateDistance(i_x, i_y, j_x, j_y)
        
        if dist < 30:
          joints2.append([i,j])   

    #Draw lines between those points
    for points in joints2:
      x_1, y_1 = points[0].ravel()
      x_2, y_2 = points[1].ravel()
      cv2.line(img2, (x_1, y_1), (x_2, y_2), (150, 0, 0), thickness=4, lineType=8)

    mask_joints2 = cv2.inRange(img2, np.array([50,0,0]), np.array([250,0,0]))

    #generate kernel for morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    
    closing_joints2 = cv2.morphologyEx(mask_joints2, cv2.MORPH_CLOSE, kernel)
    
    ret2, thresh2 = cv2.threshold(closing_joints2, 127, 255, 0)
    
    contours_joints2, hierarchy2 = cv2.findContours(thresh2, 1, 2) 
    
    if len(contours_joints2) == 4:
      for contour in contours_joints2:
        ((new_x, new_z), radius) = cv2.minEnclosingCircle(contour)

        joint_number = None
        smallest_distance = 1000

        for num, joint in enumerate(self.previous_joints):
          joint_distance = self.calculateDistance(new_x, new_z, joint[0], joint[2])

          if joint_distance < smallest_distance:
            smallest_distance = joint_distance
            joint_number = num
        
        self.previous_joints[joint_number][0] = new_x
        #self.previous_joints[joint_number][2] = new_z

        #print(joint_number, ",", smallest_distance)

  def joint_state_estimation_4_3(self):
    ######################################################################
    #Joint state estimation for task 3.2
    ######################################################################   
    self.move_joints_4_3()
    self.update_camera_1()
    self.update_camera_2()

    ja2 = self.detect_joint_angle_2_black(self.previous_joints[1], self.previous_joints[2])
    ja3 = self.detect_joint_angle_3_black(self.previous_joints[1], self.previous_joints[2])
    ja4 = self.detect_joint_angle_4_black(self.previous_joints[1], self.previous_joints[2], self.previous_joints[3])
   
    #publish joint positions as observed
    self.joint2_vision_pub.publish(ja2)
    self.joint3_vision_pub.publish(ja3)
    self.joint4_vision_pub.publish(ja4)

    #Display OpenCV image
    """
    for i in corners1:
        x,y = i.ravel()
        cv2.circle(img1,(x,y),3,150,-1)
    plt.imshow(img1)
    plt.show(block=False)
    plt.pause(30)
    plt.close()
    
    for i in corners2:
        x,y = i.ravel()
        cv2.circle(img2,(x,y),3,150,-1)
    plt.imshow(img2)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    """
    ######################################################################
    ######################################################################
    ######################################################################

  def robot_clock_tick(self):
    #uncomment to have the reults of the related task
    ### the relevant topics for the task should also be uncommented ###

    #Task 1
    self.move_joints_2_1()
    #joints = self.get_joint_state_2_1()
    #self.publish_joints_vision(joints)
    #self.detect_targets_2_2(from_base=True)

    #Task 2
    #self.forward_kinematics_3_1()
    #self.closed_loop_control_3_2()

    #Task 3
    self.joint_state_estimation_4_3()

  def joint_states_callback(self,data):
    #reading the joints values from the topic
    self.joints_ros = np.array([data.position[0], data.position[1], data.position[2], data.position[3]])

  def callback1(self,data):
  # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    #uncomment to save the image or to show it
    #cv2.imwrite('image_copy1.png', self.cv_image1)
    #im1=cv2.imshow('window1', self.cv_image1)
    #cv2.waitKey(1)

  def callback2(self,data):
  # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    '''
    cv2.imwrite('image_copy2.png', self.cv_image2)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)
    '''
    self.robot_clock_tick()

    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      
      #publish joint position according to sinusoidal trend or to output of closed_loop
      #Task 2.1 and 3.2
      #self.robot_joint1_pub.publish(self.joint1)
      self.robot_joint2_pub.publish(self.joint2)
      self.robot_joint3_pub.publish(self.joint3)
      self.robot_joint4_pub.publish(self.joint4)

      #Task 2.1
      #publish the joint position calculated using vision
      #self.joint1_vision_pub.publish(self.joint1_vision)
      #self.joint2_vision_pub.publish(self.joint2_vision)
      #self.joint3_vision_pub.publish(self.joint3_vision)
      #self.joint4_vision_pub.publish(self.joint4_vision)

      #Task 2.2
      #publish the target position calculated using vision
      #self.sphere_target_x_pub.publish(self.sphere_x)
      #self.sphere_target_y_pub.publish(self.sphere_y)
      #self.sphere_target_z_pub.publish(self.sphere_z)

      #Task 3.2
      #Publish the position of the end effector output of the closed loop
      #self.xe_pub.publish(self.xe)
      #self.ye_pub.publish(self.ye)
      #self.ze_pub.publish

      #Task 4
      #Publish the joints of the all-black robot
      #joints are published within joint_state_estimation_4_3

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


