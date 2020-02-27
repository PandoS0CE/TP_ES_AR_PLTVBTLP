import numpy as np
import cv2 as cv

class ARPipeline:
    def __init__(self,escapeKey='q',width=640,height=480,video=0,loop=True,realMode=False):
        self.escapeKey=escapeKey
        self.width=width
        self.height=height
        self.video=video
        self.loop=loop
        self.realMode=realMode
        print('[INFO] Press \"'+str(self.escapeKey)+'\" to quit')

        if video==0 :
            self.cam = cv.VideoCapture(video,cv.CAP_DSHOW)
            print('[INFO] Video capture is \"camera\"')
        else :
            self.cam = cv.VideoCapture(video)
            self.video=video

            print('[INFO] Video capture is \"'+str(video)+'\", '+str(int(self.cam.get(cv.CAP_PROP_FPS)))+' FPS')
        
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
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
        self.marker_keypoints, self.marker_descriptor = self.descriptor_extractor.detectAndCompute(cv.cvtColor(self.GetMarker(), cv.COLOR_BGR2GRAY), None)
        print('[INFO] Marker used is \"'+str(self.markerPath)+'\"')
    
    def GetMarker(self):
        return self.marker

    def GetMarkerKeypoints(self):
        return self.marker_keypoints

    def GetFrame(self):
        # Capture frame-by-frame
        if self.cam.isOpened() :
            ret, frame = self.cam.read()

            if self.video !=0 :
                # Video from file
                if not(self.realMode==True) :
                    for i in range(int(self.cam.get(cv.CAP_PROP_FPS))) :
                        if  cv.waitKey(1) & 0xFF == ord(str(self.escapeKey)):
                            # Stop with key
                            self.cam.release()
                            cv.destroyAllWindows()
                            quit()

                if not ret :
                    if self.loop==True :
                        self.cam.set(cv.CAP_PROP_POS_FRAMES, 0)
                        return self.GetFrame()
                    else :
                        # End of video file
                        self.cam.release()
                        cv.destroyAllWindows()
                        quit()
                else:
                    # resize video frame
                    frame = cv.resize(frame,(self.width,self.height))
            else :
                # Video from camera check return
                if not ret :
                    print('[ERROR] Unable to capture video')
                    self.cam.release()
                    cv.destroyAllWindows()
                    quit()    

            if  cv.waitKey(1) & 0xFF == ord(str(self.escapeKey)):
                # Stop with key
                self.cam.release()
                cv.destroyAllWindows()
                quit()
        else :
            print('[ERROR] Unable to capture video source')
            self.cam.release()
            cv.destroyAllWindows()
            quit()

        return frame
    
    def ComputeMatches(self,frame,minMatches=10):
        good = []
        frame_gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.descriptor_extractor.detectAndCompute(frame_gray, None)
        
        matches=self.matcher.knnMatch(frame_des,self.marker_descriptor,k=2)
        if len(matches) > minMatches :
            # Enough matches to apply ratio test
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([n])

        return good,frame_kp

    def ComputeHomography(self,matches,frame_kp):
        homography=None
        mask=None
        
        if len(matches) >0 :      
            src_points = np.float32([frame_kp[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_points = np.float32([self.GetMarkerKeypoints()[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)
            homography, mask=cv.findHomography(src_points,dst_points,cv.RANSAC,ransacReprojThreshold=3.0)
        return homography,mask
    
    def RefineMatches(self,matches,frame_kp):
        correct_matches=[]
        _,mask=self.ComputeHomography(matches,frame_kp)
        
        # Get inliers mask
        correct_matches = [matches[i] for i in range(len(matches)) if mask[i]]

        return correct_matches
    
    def FindMarker(self,frame,homography,minMatches=10):
        found=False
        #if homography is not None :
            # Warp marker image using homography
            # warped=cv.warpPerspective(src=cv.cvtColor(self.GetMarker(), cv.COLOR_BGR2GRAY), M=homography,dsize=(frame.shape[0],frame.shape[1]),flags=cv.WARP_INVERSE_MAP | cv.INTER_CUBIC)  
            # #matches,wrapped_kp_=self.ComputeMatches(warped,minMatches)
            # cv.imshow('test',warped)
        return found

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

def DrawRectangle(img,marker,homography):
    frame=img.copy()
    if homography is not None :
        # Draw a rectangle that marks the found model in the frame
        h=marker.shape[0]
        w=marker.shape[1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv.perspectiveTransform(pts, homography)
        # connect them with lines  
        frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    return frame  


 