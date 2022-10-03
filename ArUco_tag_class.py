import numpy as np
from networktables import NetworkTables
import cv2
import cv2.aruco as aruco
import time
import math
from ArUco_tag_main import matrix,roborio,coordinate_info,marker

matrix = matrix()
roborio = roborio()
coordinate_info = coordinate_info()
marker = marker()

if __name__ == '__main__': 
    
    roborio.connection()
    
    c_in_w = np.array([],dtype="double")
   
    rotation_vector = np.zeros((3,1))
    
    translation_vector = np.zeros((3,1)) 
    

    
    while(True):
        frame,ids,corners0 = marker.draw()
        
        if np.all(ids != None):
            
            coordinate_info.detect_ids(ids.size,ids)
            
            pts_dst = coordinate_info.sort_corner(corners0,ids.size,frame)
            
            objPoints_in = coordinate_info.sort_object(coordinate_info.objPoints,ids.size)

            print(objPoints_in)

            print(pts_dst)
            (success, rotation_vector, translation_vector) = cv2.solvePnP(objPoints_in,pts_dst, matrix.camera_matrix, None)
            
            frame = aruco.drawAxis(frame,matrix.camera_matrix,matrix.dist_coeffs,rotation_vector,translation_vector,0.1)
            
            c_in_w,facing = coordinate_info.camera_point(rotation_vector,translation_vector)
            
            trans = [facing[0][0],facing[1][0],facing[2][0]]
            
            trans=np.delete(trans,2)
            
            theta = matrix.included_angle(trans)
            
            aver_c_in_w = coordinate_info.average(c_in_w)
            
            roborio.point_info(c_in_w,theta)
            
            if aver_c_in_w != 0:
                print("{:.1f}".format(aver_c_in_w[0]), end=" ")
                print("{:.1f}".format(aver_c_in_w[1]), end=" ")
                print("{:.1f}".format(aver_c_in_w[2]),)
                
            
            coordinate_info.detect_clear()
        cv2.imshow('Display',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

