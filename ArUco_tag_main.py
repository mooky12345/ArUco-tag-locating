import numpy as np
from networktables import NetworkTables
import cv2
import cv2.aruco as aruco
import math
#import openpyxl as xl
subPixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
a = 0.25
b = -0.25
#wb = xl.load_workbook('at_home.xlsx')
#ws = wb.active
class matrix:
    
    def __init__(self):
        self.v = np.array([0,0,0])
        self.y_vector = [1,0]
        self.camera_matrix = np.array(
                            [[829.5010500,0,312.263367],
                            [0,828.729047,207.158177],
                            [0, 0, 1]], dtype = np.float64)
        self.dist_coeffs = np.zeros((5,1)) 
    def projection(self,arr):
        v_norm = np.sqrt(sum(self.v**2))
        return (np.dot(arr,v)/v_norm**2)*v
    
    def included_angle(self,arr1):
        angle = math.atan(arr1[1]/arr1[0])
        return angle
class coordinate_info:
    
    def __init__(self):
        self.aver_x = 0 
        self.aver_y = 0 
        self.aver_z = 0
        self.detect = []
        self.carmera_original_point = np.array([[0],
                                    [0],
                                    [0]],dtype="double")
        self.carmera_forward_point = np.array([[1],
                                    [0],
                                    [0]],dtype="double")
        self.objPoints = np.array([[-0.042,0.03,0.03],
                        [-0.042,-0.03,0.03],
                        [-0.042,-0.03,-0.03],
                        [-0.042,0.03,-0.03],#0
                        [-0.03,-0.042,0.03],
                        [0.03,-0.042,0.03],
                        [0.03,-0.042,-0.03],
                        [-0.03,-0.042,-0.03],#1
                        [0.042,-0.03,0.03],
                        [0.042,0.03, 0.03],
                        [0.042,0.03,-0.03],
                        [0.042,-0.03,-0.03],#2
                        [0.03,0.042,0.03],
                        [-0.03,0.042,0.03],
                        [-0.03,0.042,-0.03],
                        [0.03,0.042,-0.03],#3
                        [a-0.042,b+0.03,0.03],
                        [a-0.042,b-0.03,0.03],
                        [a-0.042,b-0.03,-0.03],
                        [a-0.042,b+0.03,-0.03],#0
                        [a-0.03,b-0.042,0.03],
                        [a+0.03,b-0.042,0.03],
                        [a+0.03,b-0.042,-0.03],
                        [a-0.03,b-0.042,-0.03],#1
                        [a+0.042,b-0.03,0.03],
                        [a+0.042,b+0.03, 0.03],
                        [a+0.042,b+0.03,-0.03],
                        [a+0.042,b-0.03,-0.03],#2
                        [a+0.03,b+0.042,0.03],
                        [a-0.03,b+0.042,0.03],
                        [a-0.03,b+0.042,-0.03],
                        [a+0.03,b+0.042,-0.03],##7
                        ], dtype=np.float64)
        self.count = 0
    def detect_ids(self,ids_size,ids):
        for i in range(ids_size):
            self.detect.append(ids[i][0])
#     def read_excel():
#         a1 = [[ws.rows[0][2],ws.rows[0][3],ws.rows[0][4]]]
#         self.objPoints = np.array(a1)
#         for i in ws.rows:
#             if i == 0:
#                 for x in range(5,12,3):
#                     for j in row[x,x+3]:
#                         self.objPoints = np.append(objPoints,j,axis=0)
#             else:
#                 for x in range(2,12,3):
#                     for j in row[x,x+3]:
#                         self.objPoints = np.append(objPoints,j,axis=0)
#         print(self.objPoints)
    def detect_clear(self):
        self.detect.clear()
        
    def sort_corner(self,corners0,ids,frame):
        a = 0
        grayImg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x1_left_down = [corners0[0][0][0][0], corners0[0][0][0][1]]
        x2_right_down = [corners0[0][0][1][0], corners0[0][0][1][1]]
        x3_left_up = [corners0[0][0][2][0], corners0[0][0][2][1]]
        x4_right_up = [corners0[0][0][3][0], corners0[0][0][3][1]]
        pts_dst = np.array([x1_left_down,x2_right_down, x3_left_up,x4_right_up])
        pts_dst = cv2.cornerSubPix(grayImg,pts_dst, (5,5), (-1,-1), subPixCriteria)
        for i in range(1,ids):
            x1_left_down = [corners0[i][0][0][0], corners0[i][0][0][1]]
            x2_right_down = [corners0[i][0][1][0], corners0[i][0][1][1]]
            x3_left_up = [corners0[i][0][2][0], corners0[i][0][2][1]]
            x4_right_up = [corners0[i][0][3][0], corners0[i][0][3][1]]
            pts_dst1 = np.array([x1_left_down,x2_right_down, x3_left_up,x4_right_up])
            pts_dst = np.append(pts_dst,cv2.cornerSubPix(grayImg,pts_dst1, (5,5), (-1,-1), subPixCriteria),axis=0)
                     

                         
        return pts_dst
    def sort_object(self,objPoints,ids):
        objPoints_in = np.array(objPoints[self.detect[0]*4:self.detect[0]*4+4])
        for i in range(1,ids):
            objPoints_in = np.append(objPoints_in,objPoints[self.detect[i]*4:self.detect[i]*4+4],axis=0)
        return objPoints_in
    
    def camera_point(self,rotation,translation):
        R_back, _ = cv2.Rodrigues(rotation)
        invert_R = np.transpose(R_back)
        R = self.carmera_original_point - translation
        R_F = self.carmera_forward_point - translation
        c_in_w = np.dot(invert_R,R)
        c_front_in_w = np.dot(invert_R, R_F)
        facing = c_front_in_w - c_in_w
        print(c_in_w)
        
        return c_in_w,facing
    
    def average(self,c_in_w):
        print(c_in_w)
        
        self.aver_x += c_in_w[0][0]
        self.aver_y += c_in_w[1][0]
        self.aver_z += c_in_w[2][0]
       
        self.count += 1
        print(self.count)
        if self.count == 3:
            average = [self.aver_x/3*100,self.aver_y/3*100,self.aver_z/3*100]
            self.aver_x = 0 
            self.aver_y = 0 
            self.aver_z = 0
            self.count = 0
            return average
        else:
            return 0
        
class roborio:
    
    def __init__(self):
        
        self.roborio_ip = 'roborio-7589-frc.local'
        self.sd = NetworkTables.getTable('SmartDashboard')
    def connection(self):
        NetworkTables.initialize(server=self.roborio_ip)
       
        
    def point_info(self,c_in_w,theta):
        
        self.sd.putNumber('cord_x', float(c_in_w[0]))
        self.sd.putNumber('cord_y', float(c_in_w[1]))
        self.sd.putNumber('cord_z', float(c_in_w[2]))
        self.sd.putNumber('cord_sita', float(theta))
            
class camera:
    def __init__(self):
       None
       
class marker:
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
    def draw(self):
        ret, frame = self.cap.read()
        
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        arucoParameters = aruco.DetectorParameters_create()

        corners0, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=arucoParameters)

        frame = aruco.drawDetectedMarkers(frame, corners0,ids)
        return frame,ids,corners0

   