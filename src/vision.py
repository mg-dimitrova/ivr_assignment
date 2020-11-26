#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

from matplotlib import pyplot as plt

import math

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
    self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    # initialize the time variables
    self.time_joints = rospy.get_time()
    #self.time_joint4 = rospy.get_time()
    #initiate publishers for joints' angular position estimated from camera 1 and 2
    #self.joint1_cam1_pub = rospy.Publisher("/joint1_camera1",Float64, queue_size=10)
    self.joint1_cam1_pub = rospy.Publisher("/joint1",Float64, queue_size=10)
    self.joint2_cam1_pub = rospy.Publisher("/joint2",Float64, queue_size=10)
    self.joint3_cam1_pub = rospy.Publisher("/joint3",Float64, queue_size=10)
    self.joint4_cam1_pub = rospy.Publisher("/joint4",Float64, queue_size=10)
    self.previous_joints = np.array([0.0, 0.0, 0.0, 0.0], dtype ='float64')
    #self.joints_cam1_pub = rospy.Publisher("/joints_cam1", Float64MultiArray, queue_size=10)
    self.error_joint_pub = rospy.Publisher("/error_joint", Float64, queue_size=10)
    self.error_joint = 0.0
    #publishers for the target coordinates
    self.sphere_target_x_pub = rospy.Publisher("/sphere_x", Float64, queue_size=10)
    self.sphere_target_y_pub = rospy.Publisher("/sphere_y", Float64, queue_size=10)
    self.sphere_target_z_pub = rospy.Publisher("/sphere_z", Float64, queue_size=10)
    #publishers for the end_effector position calculated with vision
    self.end_effector_vision_pub = rospy.Publisher("/end_effector_vision", Float64MultiArray, queue_size=10)
    self.end_effector_fk_pub = rospy.Publisher("/end_effector_fk", Float64MultiArray, queue_size=10)
    #initialize error
    self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
    self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
    self.error = np.array([0.0,0.0,0.0], dtype ='float64')
    self.error_d = np.array([0.0,0.0,0.0], dtype ='float64')

    self.initialize_joints()

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

  

  def detect_individual_joint_angles(self, image1, image2):
    #assuming joint one does not rotate
    '''
    joint1_cam1, joint1_cam2 = self.previous_joints[0], self.previous_joints[0]
    joint2_cam1, joint2_cam2 = self.previous_joints[1], self.previous_joints[1]
    joint3_cam1, joint3_cam2 = self.previous_joints[2], self.previous_joints[2]
    joint4_cam1, joint4_cam2 = self.previous_joints[3], self.previous_joints[3]
    '''
    joint1 = self.previous_joints[0]
    joint2 = self.previous_joints[1]
    joint3 = self.previous_joints[2]
    joint4 = self.previous_joints[3]
    #detecting the objects using colour recognition
    flagBlue1, circleBlue1 = detect_colour(image1, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    flagBlue2, circleBlue2 = detect_colour(image2, self.BLUE_LOWER, self.BLUE_UPPER)
    flagGreen1, circleGreen1 = detect_colour(image1, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    flagGreen2, circleGreen2 = detect_colour(image2, self.GREEN_LOWER, self.GREEN_UPPER)
    flagRed1, circleRed1 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER) #End effector
    flagRed2, circleRed2 = detect_colour(image2, self.RED_LOWER, self.RED_UPPER)
    #calculating the joints
    joint2_cam1 = np.arctan2((circleBlue1[0] - circleGreen1[0]), (circleBlue1[1] - circleGreen1[1]))
    joint2_cam2 = -np.arctan2((circleBlue2[0] - circleGreen2[0]), (circleBlue2[1] - circleGreen2[1]))
    joint3_cam1 = np.arctan2((circleBlue1[0] - circleGreen1[0]), (circleBlue1[1] - circleGreen1[1]))
    joint3_cam2 = -np.arctan2((circleBlue2[0] - circleGreen2[0]), (circleBlue2[1] - circleGreen2[1]))
    joint4_cam1 = np.arctan2((circleGreen1[0] - circleRed1[0]),(circleGreen1[1] - circleRed1[1]))
    joint4_cam2 = -np.arctan2((circleGreen2[0] - circleRed2[0]),(circleGreen2[1] - circleRed2[1]))
    if flagBlue1 != 1 and flagGreen1 != 1:
      print(joint2_cam1, joint2_cam2)
      joint2 = joint2_cam1
      self.previous_joints[1] = joint2
    elif flagBlue2 != 1 or flagGreen2 != 1:
      print(joint2_cam1, joint2_cam2)
      joint2 = joint2_cam2
      self.previous_joints[1] = joint2
    if flagBlue2 != 1 or flagGreen2 != 1:
      #from camera 2 the x axis increases going left, opposite of the reference system of the robot
      joint3 = np.arctan2((circleGreen2[0] - circleBlue2[0]), (circleBlue2[1] - circleGreen2[1]))
      self.previous_joints[2] = joint3
    elif flagBlue1 != 1 or flagGreen1 != 1:
      joint3 = np.arctan2((circleBlue1[0] - circleGreen1[0]), (circleBlue1[1] - circleGreen1[1]))
      self.previous_joints[2] = joint3
    if flagGreen1 != 1 or flagRed1 != 1:
      joint4 = np.arctan2((circleGreen1[0] - circleRed1[0]),(circleGreen1[1] - circleRed1[1])) #- joint2_cam1 - joint3_cam2
      self.previous_joints[3] = joint4
    elif flagGreen2 != 1 or flagRed2 != 1:
      joint4= np.arctan2((circleRed2[0] - circleGreen2[0]),(circleGreen2[1] - circleRed2[1])) #- joint2_cam2 - joint3_cam2
      self.previous_joints[3] = joint4

    #return np.array([joint1_cam1, joint2_cam1, joint3_cam1, joint4_cam1]), np.array([joint1_cam2, joint2_cam2, joint3_cam2, joint4_cam2])
    return np.array([joint1, joint2, joint3, joint4])

  def detect_joints_3D(self, image1, image2, assume_zero=False, previous_state=False, predict=False):
    #detect joints from camera1
    #flag_center, center = detect_colour(image1, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    yellow = np.array([398,398,532])
    circleBlue1 = detect_colour(image1, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    circleBlue2 = detect_colour(image2, self.BLUE_LOWER, self.BLUE_UPPER)
    circleGreen1 = detect_colour(image1, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    circleGreen2 = detect_colour(image1, self.GREEN_LOWER, self.GREEN_UPPER)
    circleRed1 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER) #End effector
    circleRed2 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER)
    #joint 1 is assumed not to be changing for task 2.1, thus joint 3 can be detected only from camera2
    #we assume joint1 is not rotating for task 2.1
    #  self.joint1_cam1 = np.arctan2((center[0] - circleBlue1[0]), (center[1] - circleBlue1[1]))

    blue = np.array([circleBlue2[0], circleBlue1[0], circleBlue1[1]])
    green = np.array([circleGreen2[0], circleGreen1[0], circleGreen2[1]])
    red = np.array([circleRed2[0], circleRed1[0], circleRed2[1]])
    '''
    #Blue
    a = np.array([circleBlue2[0], circleBlue1[0], circleBlue1[1]])
    #Green
    b = np.array([circleGreen2[0], circleGreen1[0], circleGreen2[1]])
    #Red
    c = np.array([circleRed2[0], circleRed1[0], circleRed2[1]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    #print(np.degrees(angle))

    '''
    norm_yb = blue - yellow
    norm_bg = green - blue
    norm_gr = red - green
    #print(np.linalg.norm(norm_yb), np.linalg.norm(norm_bg), np.linalg.norm(norm_gr))
    cosine_j4 = np.dot(norm_bg, norm_gr) / (np.linalg.norm(norm_bg) * np.linalg.norm(norm_gr))
    j4 = np.arccos(cosine_j4)
    #self.joint4_cam1 = j4
    cosine_j3 = np.dot(norm_yb, norm_bg) / (np.linalg.norm(norm_yb) * np.linalg.norm(norm_bg))
    j3 = np.arccos(cosine_j3)
    #self.joint2_cam1 = j3[0]
    #self.joint3_cam1 = j3[1]
    #print(j3, j4)

  def detect_joints_3D_2(self, camera1_perspective, camera2_perspective, joint1, joint2, joint3):
    joint1_camera1 = detect_colour(camera1_perspective, joint1[0], joint1[1])
    joint1_camera2 = detect_colour(camera2_perspective, joint1[0], joint1[1])
    joint2_camera1 = detect_colour(camera1_perspective, joint2[0], joint2[1])
    joint2_camera2 = detect_colour(camera2_perspective, joint2[0], joint2[1])
    joint3_camera1 = detect_colour(camera1_perspective, joint3[0], joint3[1])
    joint3_camera2 = detect_colour(camera2_perspective, joint3[0], joint3[1])

 
    a = np.array([joint1_camera2[0], joint1_camera1[0], joint1_camera2[1]])
    b = np.array([joint2_camera2[0], joint2_camera1[0], joint2_camera2[1]])
    c = np.array([joint3_camera2[0], joint3_camera1[0], joint3_camera2[1]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    #print(np.degrees(angle))

  def detect_joint_angle_2_blue(self, image1, image2):
    circleBlue = self.detect_colour2(image1, image2, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    circleGreen = self.detect_colour2(image1, image2, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    theta = np.arcsin((circleGreen[1] - circleBlue[1])/ 100) #Max length for link
    return theta

  def detect_joint_angle_3_blue(self, image1, image2):
    circleBlue = self.detect_colour2(image1, image2, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    circleGreen = self.detect_colour2(image1, image2, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    theta = np.arcsin((circleGreen[0] - circleBlue[0])/ 100) #Max length for link
    return theta

  def detect_joint_angle_4_green(self, image1, image2):
    circleBlue = self.detect_colour2(image1, image2, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    circleGreen = self.detect_colour2(image1, image2, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    circleRed = self.detect_colour2(image1, image2, self.RED_LOWER, self.RED_UPPER) #End effector
    
    #Blue
    a = circleBlue
    #Green
    b = circleGreen
    #Red
    c = circleRed

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    #return np.degrees(angle)
    return angle
  
  def detect_end_effector(self, image1, image2):
    yellow = np.array([398, 532])
    #there is a little offset on z, but the x and y coordinates calculated by the cameras are the same +- 1pixel
    flagRed1, circleRed1 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER) #End effector
    flagRed2, circleRed2 = detect_colour(image2, self.RED_LOWER, self.RED_UPPER)
    #joint 1 is assumed not to be changing for task 2.1, thus joint 3 can be detected only from camera2
    r_yz = self.PIXEL2METER*(yellow-circleRed1)
    r_xz = self.PIXEL2METER*(yellow-circleRed2)
    #print(r_xz, r_yz)
    return np.array([r_xz[0], r_yz[0], r_yz[1]])


  def pixel2meter(self, image):
    #this value is always the same... we should not calculate it all the time... =
    # dist = 0.03845698760800996 m/px dist2 = 0.03888648856244221 m/px
    #link1= 64 px link2 = 90 px
    #we calculate the length of the link from link 1 as it is the only one whose joint angle does not change
    self.p2m = Float64()
    self.p2m2 = Float64()
    self.previous_p2m = Float64()
    flag_center, center = detect_colour(image, self.YELLOW_LOWER, self.YELLOW_UPPER)
    flagBlue1, circleBlue1 = detect_colour(image, self.BLUE_LOWER, self.BLUE_UPPER)
    flagGreen1, circleGreen1 = detect_colour(image, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    dist = np.sum((center - circleBlue1)**2)
    dist2 = np.sum((circleBlue1 - circleGreen1)**2)
    if flagBlue1 != 1 and flag_center != 1:
      self.p2m = 2.5/np.sqrt(dist)
      self.previous_p2m = self.pixel2meter
      self.p2m2 = (3.5/np.sqrt(dist2))
    else:
      self.p2m = self.previous_p2m


  def pixels_to_meters(self, pixels):
      return pixels / self.PIXEL_PER_METER

  def forward_kinematics(self,joints):
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
    #t1, t2, t3, t4 = self.joint1_cam1, self.joint2_cam1, self.joint3_cam1, self.joint4_cam1
    t1, t2, t3, t4 = joints[0], joints[1], joints[2], joints[3]
    #l1 = 2.5
    l3 = 3.5
    l4 = 3.0
    J11 = l4*np.cos(t4)*(np.cos(t1)*np.sin(t2)*np.cos(t3) - np.sin(t1)*np.sin(t3)) + l4*np.sin(t4)*np.cos(t1)*np.cos(t2) + l3*np.cos(t3)*np.cos(t1)*np.sin(t2) - l3*np.sin(t1)*np.sin(t3)
    J21 = l4*np.cos(t4)*(np.sin(t1)*np.sin(t2)*np.cos(t3) + np.cos(t1)*np.sin(t3)) + l4*np.sin(t4)*np.sin(t1)*np.cos(t2) + l3*np.cos(t3)*np.sin(t1)*np.sin(t2) + l3*np.cos(t1)*np.sin(t3)
    J31 = 0.0
    J12 = l4*np.cos(t4)*(np.sin(t1)*np.cos(t2)*np.cos(t3)) - l4*np.sin(t4)*np.sin(t1)*np.sin(t2) + l3*np.cos(t3)*np.sin(t1)*np.cos(t2)
    J22 = -l4*np.cos(t4)*np.cos(t1)*np.cos(t2)*np.cos(t3) + l4*np.sin(t4)*np.cos(t1)*np.sin(t2) - l3*np.cos(t3)*np.cos(t1)*np.cos(t2)
    J32 = -l4*np.cos(t4)*np.sin(t2)*np.cos(t3) - l4*np.sin(t4)*np.cos(t2) - l3*np.cos(t3)*np.sin(t2)
    J13 = l4*np.cos(t4)*(np.sin(t1)*np.sin(t2)*np.cos(t3) + np.cos(t1)*np.sin(t3)) + l4*np.sin(t4)*np.sin(t1)*np.cos(t2) + l3*np.cos(t3)*np.sin(t1)*np.sin(t2) + l3*np.cos(t1)*np.sin(t3)
    J23 = l4*np.cos(t4)*(-np.cos(t1)*np.sin(t2)*np.cos(t3) + np.sin(t1)*np.sin(t3)) - l4*np.sin(t4)*np.cos(t1)*np.cos(t2) - l3*np.cos(t3)*np.cos(t1)*np.sin(t2) + l3*np.sin(t1)*np.sin(t3)
    J33 = -l4*np.cos(t4)*np.cos(t2)*np.sin(t3) - l3*np.sin(t3)*np.cos(t2)
    J14 = -l4*np.sin(t4)*(np.sin(t1)*np.sin(t2)*np.cos(t3) + np.cos(t1)*np.sin(t3)) + l4*np.cos(t4)*np.sin(t1)*np.cos(t2)
    J24 = -l4*np.sin(t4)*(-np.cos(t1)*np.sin(t2)*np.cos(t3) + np.sin(t1)*np.sin(t3)) - l4*np.cos(t4)*np.cos(t1)*np.cos(t2)
    J34 = -l4*np.sin(t4)*np.cos(t2)*np.cos(t3) - l4*np.cos(t4)*np.sin(t2)
    J = np.array([[J11, J12, J13, J14],[J21, J22, J23, J24], [J31, J32, J33, J34]])
    return J

  def calculate_pseudo_invert_jacobian(self,joints):
    #check if the inv matrix has to be changed in 3D -> jacobian is 3x4
    J = self.calculate_jacobian(joints)
    pinv_J = np.linalg.pinv(J)
    return pinv_J

  def control_open(self, image, joints):
    invJ = self.calculate_pseudo_invert_jacobian(joints)
    current_time = rospy.get_time()
    dt = current_time - self.time_previous_step
    self.time_previous_step = current_time
    x_d = self.detect_targets(self.cv_image1, self.cv_image2, sphere=True)
    self.error_d = (x_d - self.error)/dt
    self.error = x_d
    qdot = np.dot(invJ,self.error_d.transpose())
    q = joints + dt * qdot
    return q

  def closed_loop(self, image1, image2, joints):
    Kp = np.array([[0.1, 0],[0,0.1]])
    Kd = Kp
    invJ = self.calculate_pseudo_invert_jacobian(joints)
    x_d = self.detect_targets(image1, image2, sphere=True)
    x_e = self.detect_end_effector(image1, image2)
    err = x_d - x_e
    current_time = rospy.get_time()
    dt = current_time - self.time_previous_step2
    self.time_previous_step2 = current_time
    self.error_d = (err - self.error)/dt
    self.error = err
    qdot = np.dot(invJ, (np.dot(Kp, self.error.transpose()) + np.dot(Kd, self.error_d.transpose())))
    q = joints + dt * qdot
    return q


  def detect_colour2(self, image1, image2, lower_colour_boundary, upper_colour_boundary, target=None):
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

  def detect_targets(self, image1, image2, cube=False, sphere=True):
    contours1 = detect_colour(image1, self.ORANGE_LOWER, self.ORANGE_UPPER, is_target = True)
    contours2 = detect_colour(image2, self.ORANGE_LOWER, self.ORANGE_UPPER, is_target = True)
    #_, center_white1 = detect_colour(image1, self.WHITE_LOWER, self.WHITE_UPPER)
    #flag_center1, center_yz = detect_colour(image1, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    #flag_center2, center_xz = detect_colour(image2, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    #center_yz = np.array([398.0,532.0]) #yellow joint
    #center_xz = np.array([398.0,533.0])
    #center_yz = np.array([398.0,545.0])
    #center_xz = center_yz
    
    target_coordinates_x = 0
    target_coordinates_y = 0
    target_coordinates_z = 0
    
    if len(contours1) != 0:
      for contour in contours1:
        if cube and is_cube(contour):
            M = cv2.moments(contour)
            if M['m00'] != 0:
              cube_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
              self.cube_y = self.PIXEL2METER*(cube_coords[0] - center_yz[0])
              self.cube_z = self.PIXEL2METER*(center_yz[1] - cube_coords[1])
        elif sphere and is_sphere(contour):
            M = cv2.moments(contour)
            if M['m00'] != 0:
              sphere_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
              target_coordinates_y = sphere_coords[0]
              target_coordinates_z = sphere_coords[1]

    if len(contours2) != 0:
      for contour in contours2:
        if cube and is_cube(contour):
            M = cv2.moments(contour)
            if M['m00'] != 0:
              cube_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
              self.cube_x = self.PIXEL2METER*(cube_coords[0]- center_xz[0])
              self.cube_z = self.PIXEL2METER*(center_xz[1] - cube_coords[1])
        if sphere and is_sphere(contour):
            M = cv2.moments(contour)
            if M['m00'] != 0:
              sphere_coords = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
              target_coordinates_x = sphere_coords[0]

      return [target_coordinates_x, target_coordinates_y, target_coordinates_z]

  def get_target_x(self, source_point, target_point):
    return self.pixels_to_meters(target_point[0] - source_point[0])

  def get_target_y(self, source_point, target_point):
    return self.pixels_to_meters(target_point[1] - source_point[1])

  def get_target_z(self, source_point, target_point):
    return self.pixels_to_meters(source_point[2] - target_point[2] + 0.8) #Base Offset

  def detect_target_range(self, source_point, target_point):    
    print("Target X: ", self.get_target_x(source_point, target_point))
    print("Target Y: ", self.get_target_y(source_point, target_point))
    print("Target Z: ", self.get_target_z(source_point, target_point))


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
    self.robot_joint2_pub.publish(self.joint2)
    self.robot_joint3_pub.publish(self.joint3)
    self.robot_joint4_pub.publish(self.joint4)
    ######################################################################
    ######################################################################
    ######################################################################

  def get_joint_state_2_1(self):
    ######################################################################
    #Joint state estimation for task 2.1
    ######################################################################
    angles = self.detect_individual_joint_angles(self.cv_image1, self.cv_image2)
    print(angles)

    joint_2 = self.detect_joint_angle_2_blue(self.cv_image1, self.cv_image2)
    print("Joint 2:", joint_2)

    joint_3 = self.detect_joint_angle_3_blue(self.cv_image1, self.cv_image2)
    print("Joint 3:", joint_3)

    joint_4 = self.detect_joint_angle_4_green(self.cv_image1, self.cv_image2)
    print("Joint 4:", np.pi - joint_4)
    ######################################################################
    ######################################################################
    ######################################################################
    

  def detect_targets_2_2(self):
    ######################################################################
    #Detect the sphere target for task 2.2
    ######################################################################
    self.detect_target_range(self.detect_colour2(self.cv_image1, self.cv_image2, self.YELLOW_LOWER, self.YELLOW_UPPER), self.detect_colour2(self.cv_image1, self.cv_image2, self.ORANGE_LOWER, self.ORANGE_UPPER, "Sphere"))
    ######################################################################
    ######################################################################
    ######################################################################


  def forward_kinematics_3_1(self):
    ######################################################################
    #Calculate FK for task 3.1
    ######################################################################
    calculated_value = self.forward_kinematics([0.0, 0.0, 0.0, 0.3])
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
    self.closed_loop()
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
      
    curr_time = Float64()
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
    self.move_joints_4_3()
    self.update_camera_1()
    self.update_camera_2()

    ja2 = self.detect_joint_angle_2_black(self.previous_joints[1], self.previous_joints[2])
    ja3 = self.detect_joint_angle_3_black(self.previous_joints[1], self.previous_joints[2])
    ja4 = self.detect_joint_angle_4_black(self.previous_joints[1], self.previous_joints[2], self.previous_joints[3])
   
    #publish joint positions as observed
    self.joint2_cam1_pub.publish(ja2)
    self.joint3_cam1_pub.publish(ja3)
    self.joint4_cam1_pub.publish(ja4)

    #THIS STUFF SHOWS A UI. DONT MESS WITH IT
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
    print("")







  def robot_clock_tick(self):
    
    
    #self.move_joints_2_1()
    #self.get_joint_state_2_1()
    #self.detect_targets_2_2()
    #self.forward_kinematics_3_1()

    self.joint_state_estimation_4_3()
    #print("")


    #joints_cam1, joints_cam2 = self.detect_individual_joint_angles(self.cv_image1, self.cv_image2)
    #joints_cam = self.detect_individual_joint_angles(self.cv_image1, self.cv_image2)
    #self.joint2_cam1.data = joints_cam[1]
    #self.joint4_cam1.data = joints_cam[3]
    #self.joint3_cam2.data = joints_cam[2]


    #self.detect_target_range(self.detect_colour2(self.cv_image1, self.cv_image2, self.YELLOW_LOWER, self.YELLOW_UPPER), self.detect_colour2(self.cv_image1, self.cv_image2, self.ORANGE_LOWER, self.ORANGE_UPPER, "Sphere"))


  def joint_states_callback(self,data):
    #print(data.position)

    self.joint1_ros_val = data.position[0]
    self.joint2_ros_val = data.position[1]
    self.joint3_ros_val = data.position[2]
    self.joint4_ros_val = data.position[3]

    #print(self.joint1_ros_val)
    #print(self.joint2_ros_val)
    #print(self.joint3_ros_val)
    #print(self.joint4_ros_val)

  def callback1(self,data):
  # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    cv2.imwrite('image_copy1.png', self.cv_image1)
    #im1=cv2.imshow('window1', self.cv_image1)
    #cv2.waitKey(1)

  def callback2(self,data):
  # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Uncomment if you want to save the image
    cv2.imwrite('image_copy2.png', self.cv_image2)
    #im2=cv2.imshow('window2', self.cv_image2)
    #cv2.waitKey(1)

    self.robot_clock_tick()

    """
    #estimate joint angles individually
    self.joint1_cam1 = Float64()
    self.joint2_cam1 = Float64()
    self.joint3_cam2 = Float64()
    self.joint4_cam1 = Float64()
    self.joints_cam1 = Float64MultiArray()
    """


    



    

    #self.end_effector_vision = Float64MultiArray()
    
    #self.end_effector_vision.data = self.detect_end_effector(self.cv_image1, self.cv_image2)
    
    #self.end_effector_fk = Float64MultiArray()
    #self.end_effector_fk.data = self.FM()

    #self.detect_joints_3D(self.cv_image1, self.cv_image2, assume_zero=True, previous_state=False, predict=False)
    #estimate the joints angle desired from sinusoidal formulas
    
    
    
    
    
    
    
    
    
    #self.error_joint = self.joint2.data - joints_cam[1]
    
    
    
    #detect coordinate of the spherical target
    self.sphere_x = Float64()
    self.sphere_y = Float64()
    self.sphere_z = Float64()
    #self.detect_targets(self.cv_image1, self.cv_image2)

    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      
      #publish joint position according to sinusoidal trend
      #self.robot_joint2_pub.publish(self.joint2)
      #self.robot_joint3_pub.publish(self.joint3)
      #self.robot_joint4_pub.publish(self.joint4)
      #print(self.joint2, self.joint3, self.joint4)
      #publish the joint position calculated using vision
      #self.joint1_cam1_pub.publish(self.joint1_cam1)
      #self.joint2_cam1_pub.publish(self.joint2_cam1)
      #self.joint3_cam1_pub.publish(self.joint3_cam2)
      #self.joint4_cam1_pub.publish(self.joint4_cam1)
      #self.error_joint_pub.publish(self.error_joint)
      #publish the target position calculated using vision
      #self.sphere_target_x_pub.publish(self.sphere_x)
      #self.sphere_target_y_pub.publish(self.sphere_y)
      #self.sphere_target_z_pub.publish(self.sphere_z)
      #self.end_effector_fk_pub.publish(self.end_effector_fk)
      #self.end_effector_vision_pub.publish(self.end_effector_vision)
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


