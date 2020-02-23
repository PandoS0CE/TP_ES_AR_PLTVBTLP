import numpy as np
import cv2 as cv

class ARPipeline:
    def __init__(self,escapeKey='q',width=640,height=480):
        self.escapeKey=escapeKey
        print('[INFO] Press \"'+str(self.escapeKey)+'\" to quit')
        self.cam = cv.VideoCapture(0,cv.CAP_DSHOW)
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.descriptor_extractor = cv.ORB_create()
        self.matcher= cv.BFMatcher() #https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    def LoadCamCalibration(self,calibrationPath):
        self.calibrationPath=calibrationPath
        with np.load(str(self.calibrationPath)) as X:
            self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        print('[INFO] Calibration file used is \"'+str(self.calibrationPath)+'\"')
    
    def GetCamMatrix(self):
        return self.mtx
    
    def GetCamDistorsion(self):
        return self.dist

    def LoadMarker(self,markerPath):
        self.markerPath=markerPath
        self.marker=cv.imread(self.markerPath,cv.IMREAD_COLOR)
        self.marker_keypoints, self.marker_descriptor = self.descriptor_extractor.detectAndCompute(self.GetMarker(), None)
        print('[INFO] Marker used is \"'+str(self.markerPath)+'\"')
    
    def GetMarker(self):
        return self.marker

    def GetMarkerKeypoints(self):
        return self.marker_keypoints

    def GetFrame(self):
        # Capture frame-by-frame
        ret, frame = self.cam.read()
        if not ret:
            print('[ERROR] Unable to capture video')
            self.cam.release()
            cv.destroyAllWindows()
            quit()  

        if  cv.waitKey(1) & 0xFF == ord(str(self.escapeKey)):
            self.cam.release()
            cv.destroyAllWindows()
            quit()
        
        return frame
    
    def ComputeMatches(self,frame,minMatches):
        frame_kp, frame_des = self.descriptor_extractor.detectAndCompute(frame, None)
        
        if frame_des is None or len(frame_des)<minMatches :
            #print('[Warning] Not enough match between frame and marker')
            return None,None

        matches=self.matcher.knnMatch(frame_des,self.marker_descriptor,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        return good,frame_kp

    def HomographyEstimation(self,matches,frame_kp):
        if matches is None :
            return None

        #print(frame_kp[5].pt)
        #source_points = np.float32([frame_kp[m.trainIdx].pt for m in matches])

    def ComputePose(self):
        print("co")
        # voir pose_from_homography_dl => https://visp-doc.inria.fr/doxygen/camera_localization/tutorial-pose-dlt-planar-opencv.html




def DrawKeypoints(img,keypoints,width=None,height=None):
    if width == None :
        width=img.shape[1]
    if height == None :
        height=img.shape[0]

    tmp=img.copy()
    cv.drawKeypoints(img,keypoints,tmp ,color=(0,255,0), flags=0)
    size=(int(width),int(height))
    tmp = cv.resize(tmp,size)

    return tmp

def DrawMatches(img,img_kp,marker,marker_kp,matches,maxMatches=None):
    keypoints=DrawKeypoints(marker,marker_kp)
    if matches is None :
        tmp=cv.drawMatchesKnn(keypoints,marker_kp,img,img_kp,None,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        if maxMatches == None or maxMatches>len(matches)-1:
            maxMatches=len(matches)-1
            if maxMatches < 0 :
                maxMatches=0
        tmp=cv.drawMatchesKnn(keypoints,marker_kp,img,img_kp,matches[:maxMatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return tmp
   