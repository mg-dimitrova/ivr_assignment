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
      return np.array([cx, cy])

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
    self.joint2_cam1_pub = rospy.Publisher("/joint2",Float64, queue_size=10)
    self.joint3_cam1_pub = rospy.Publisher("/joint3",Float64, queue_size=10)
    self.joint4_cam1_pub = rospy.Publisher("/joint4",Float64, queue_size=10)
    #publishers for the target coordinates
    self.sphere_target_x_pub = rospy.Publisher("/sphere_x", Float64, queue_size=10)
    self.sphere_target_y_pub = rospy.Publisher("/sphere_y", Float64, queue_size=10)
    self.sphere_target_z_pub = rospy.Publisher("/sphere_z", Float64, queue_size=10)

  def position_joint2(self, current_time):
    j2 = float((np.pi/2) * np.sin(np.pi/15 * current_time))
    return j2

  def position_joint3(self, current_time):
    j3 = float((np.pi/2) * np.sin(np.pi/18 * current_time))
    return j3

  def position_joint4(self, current_time):
    j4 = float((np.pi/2) * np.sin(np.pi/20 * current_time))
    return j4

  def detect_individual_joint_angles(self, image1, image2, assume_zero=False, previous_state=False, predict=False):
    #detect joints from camera1
    #flag_center, center = detect_colour(image1, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    flagBlue1, circleBlue1 = detect_colour(image1, self.BLUE_LOWER, self.BLUE_UPPER) #Joint 2 & 3
    flagBlue2, circleBlue2 = detect_colour(image2, self.BLUE_LOWER, self.BLUE_UPPER)
    flagGreen1, circleGreen1 = detect_colour(image1, self.GREEN_LOWER, self.GREEN_UPPER) #Joint 4
    flagGreen2, circleGreen2 = detect_colour(image1, self.GREEN_LOWER, self.GREEN_UPPER)
    flagRed1, circleRed1 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER) #End effector
    flagRed2, circleRed2 = detect_colour(image1, self.RED_LOWER, self.RED_UPPER)
    #joint 1 is assumed not to be changing for task 2.1, thus joint 3 can be detected only from camera2

    #we assume joint1 is not rotating for task 2.1
    #  self.joint1_cam1 = np.arctan2((center[0] - circleBlue1[0]), (center[1] - circleBlue1[1]))
    if flagBlue1 == 1 or flagGreen1 == 1:
      #an error has occurred one of the joints is not visible, assign previous valid position to the current joint
      if assume_zero:
        self.joint2_cam1 = 0.0
      elif previous_state:
        self.joint2_cam1 = self.previous_joint2
      elif predict:
        pass
    else:
      self.joint2_cam1 = np.arctan2((circleBlue1[0] - circleGreen1[0]), (circleBlue1[1] - circleGreen1[1]))# - float(self.joint1_cam1)
      self.previous_joint2 = self.joint2_cam1

    if flagBlue2 == 1 or flagGreen2 == 1:
      #an error has occurred one of the joints is not visible, assign previous valid position to the current joint
      if assume_zero:
        self.joint3_cam1 = 0.0
      elif previous_state:
        self.joint3_cam1 = self.previous_joint3
      elif predict:
        pass
    else:
      self.joint3_cam1 = np.arctan2((circleBlue2[0] - circleGreen2[0]), (circleBlue2[1] - circleGreen2[1]))# - float(self.joint1_cam1)
      self.previous_joint3 = self.joint3_cam1

    if flagGreen1 == 1 or flagRed1 == 1:
      if assume_zero:
        self.joint4_cam1 = 0.0
      elif previous_state:
        self.joint4_cam1 = self.previous_joint4
    else:
      self.joint4_cam1 = np.arctan2((circleGreen1[0] - circleRed1[0]),(circleGreen1[1] - circleRed1[1])) - self.joint2_cam1 #- float(self.joint1_cam1)
      self.previous_joint4 = self.joint4_cam1

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
    """
    blue = np.array([circleBlue2[0], circleBlue1[0], circleBlue1[1]])
    green = np.array([circleGreen2[0], circleGreen1[0], circleGreen2[1]])
    red = np.array([circleRed2[0], circleRed1[0], circleRed2[1]])
    """
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

    print(np.degrees(angle))

    


    """
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
    """

    

  def detect_joints_3D_2(self, camera1_perspective, camera2_perspective, joint1, joint2, joint3):
    joint1_camera1 = detect_colour(camera1_perspective, joint1[0], joint1[1])
    joint1_camera2 = detect_colour(camera2_perspective, joint1[0], joint1[1])
    joint2_camera1 = detect_colour(camera1_perspective, joint2[0], joint2[1])
    joint2_camera2 = detect_colour(camera2_perspective, joint2[0], joint2[1])
    joint3_camera1 = detect_colour(camera1_perspective, joint3[0], joint3[1])
    joint3_camera2 = detect_colour(camera2_perspective, joint3[0], joint3[1])

    #Blue
    a = np.array([joint1_camera2[0], joint1_camera1[0], joint1_camera2[1]])
    #Green
    b = np.array([joint2_camera2[0], joint2_camera1[0], joint2_camera2[1]])
    #Red
    c = np.array([joint3_camera2[0], joint3_camera1[0], joint3_camera2[1]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    print(np.degrees(angle))

  def detect_targets(self, image1, image2, cube=False, sphere=True):
    contours1 = detect_colour(image1, self.ORANGE_LOWER, self.ORANGE_UPPER, is_target = True)
    contours2 = detect_colour(image2, self.ORANGE_LOWER, self.ORANGE_UPPER, is_target = True)
    #_, center_white1 = detect_colour(image1, self.WHITE_LOWER, self.WHITE_UPPER)
    #flag_center1, center_yz = detect_colour(image1, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    #flag_center2, center_xz = detect_colour(image2, self.YELLOW_LOWER, self.YELLOW_UPPER) #Joint 1
    #center_yz = np.array([398.0,532.0]) #yellow joint
    #center_xz = np.array([398.0,533.0])
    center_yz = np.array([398.0,545.0])
    center_xz = center_yz
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
              self.sphere_y = self.PIXEL2METER*(sphere_coords[0]- center_yz[0])
              self.sphere_z = self.PIXEL2METER*(center_yz[1] - sphere_coords[1])
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
              self.sphere_x = self.PIXEL2METER*(sphere_coords[0]- center_xz[0])
              if len(contours1) == 0:
                #if we cannot detect the sphere from camera 1 we can print z from camera 2 otherwise is redundant
                self.sphere_z = self.PIXEL2METER*(center_xz[1] - sphere_coords[1])

  def robot_clock_tick(self):
    #send control commands to joints for task 2.1
    curr_time = np.array([rospy.get_time() - self.time_joints])
    self.joint2 = Float64()
    self.joint2.data = self.position_joint2(curr_time)
    self.joint3 = Float64()
    self.joint3.data = self.position_joint3(curr_time)
    self.joint4 = Float64()
    self.joint4.data = self.position_joint4(curr_time)

    blue = [self.BLUE_LOWER, self.BLUE_UPPER]
    green = [self.GREEN_LOWER, self.GREEN_UPPER]
    red = [self.RED_LOWER, self.RED_UPPER]
    yellow = [self.YELLOW_LOWER, self.YELLOW_UPPER]

    self.detect_joints_3D_2(self.cv_image1, self.cv_image2, blue, green, red)

    self.detect_joints_3D_2(self.cv_image1, self.cv_image2, yellow, blue, green)


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

    #estimate joint angles individually
    self.joint2_cam1 = Float64()
    self.previous_joint2 = Float64()
    self.joint3_cam1 = Float64()
    self.previous_joint3 = Float64()
    self.joint4_cam1 = Float64()
    self.previous_joint4 = Float64()
    #self.detect_individual_joint_angles(self.cv_image1, self.cv_image2, assume_zero=True, previous_state=False, predict=False)
    #self.detect_joints_3D(self.cv_image1, self.cv_image2, assume_zero=True, previous_state=False, predict=False)

  


    #estimate the joints angle desired from sinusoidal formulas
    self.robot_clock_tick()
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
      self.joint2_cam1_pub.publish(self.joint2_cam1)
      self.joint3_cam1_pub.publish(self.joint3_cam1)
      self.joint4_cam1_pub.publish(self.joint4_cam1)
      #publish the target position calculated using vision
      #self.sphere_target_x_pub.publish(self.sphere_x)
      #self.sphere_target_y_pub.publish(self.sphere_y)
      #self.sphere_target_z_pub.publish(self.sphere_z)
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


