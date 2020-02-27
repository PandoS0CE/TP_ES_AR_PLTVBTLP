import numpy as np
import cv2 as cv

_debug=False
print('[INFO] Debug ARTools is : '+str(_debug))

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
        #tmp=cv.drawMatchesKnn(keypoints,marker_kp,img,img_kp,matches[:maxMatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        tmp=cv.drawMatchesKnn(marker,marker_kp,img,img_kp,matches[:maxMatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return tmp

def Draw3DRectangle(img,marker,homography,color=(255,0,0),old=None):
    frame=img.copy()
    if homography is not None :
        # Draw a rectangle that marks the found model in the frame
        h=marker.shape[0]
        w=marker.shape[1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv.perspectiveTransform(pts, homography)
        if old is not None :    
            old.Add(dst)
            dst=old.Mean()

        # connect them with lines  
        frame = cv.polylines(frame, [np.int32(dst)], True, color, 3, cv.LINE_AA)

    return frame  

def Draw3DCube(img, rvecs,tvecs,cam):
    frame=img.copy()
    if rvecs is not None and tvecs is not None :
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
        # # project 3D points to image plane
        # imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, cam.mtx, cam.dist)
        # imgpts = np.int32(imgpts).reshape(-1,2)
        # # draw ground floor in green
        # frame = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
        # # draw pillars in blue color
        # for i,j in zip(range(4),range(4,8)):
        #     frame = cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # # draw top layer in red color
        # frame = cv.drawContours(frame, [imgpts[4:]],-1,(0,0,255),3)

    return frame

def Log(message):
     if _debug==True :
        print(message)

class Struct(object):
    def __getattr__(self, name):
        setattr(self, name, None)

class FrameStacker:
    def __init__(self,n=15):
        self.n=n
        self.frames=[]

    def Add(self,object):
        if len(self.frames)==self.n :
            self.frames.pop()
        self.frames.insert(0,object)
    
    def Clear(self):
        self.frames=[]

    def Mean(self):
        m=self.frames[0].copy()
        # for f in self.frames :
        #     m=(m+f)/2 #@TODO pas bon

        return m

class ARPipeline:
    def __init__(self,escapeKey='q',width=640,height=480,video=0,loop=True,realMode=False):
        self.cam=Struct()
        self.marker=Struct()
        self.old=Struct()
        self.old.transformation=FrameStacker()
        self.escapeKey=escapeKey
        self.cam.width=width
        self.cam.height=height
        self.cam.video=video
        self.cam.loop=loop
        self.cam.realMode=realMode
        
        print('[INFO] Press \"'+str(self.escapeKey)+'\" to quit')

        if video==0 :
            self.cam.capture = cv.VideoCapture(video,cv.CAP_DSHOW)
            print('[INFO] Video capture is \"camera\"')
        else :
            self.cam.capture = cv.VideoCapture(video)
            self.cam.video=video

            print('[INFO] Video capture is \"'+str(video)+'\", '+str(int(self.cam.capture.get(cv.CAP_PROP_FPS)))+' FPS')
        
        self.cam.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.cam.width)
        self.cam.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.cam.height)
        self.descriptor = cv.ORB_create()
        self.matcher= cv.BFMatcher() #https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

    def LoadCamCalibration(self,calibrationPath):
        with np.load(str(calibrationPath)) as X:
            self.cam.mtx, self.cam.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        print('[INFO] Calibration file used is \"'+str(calibrationPath)+'\"')
    
    def GetCamMatrix(self):
        return self.cam.mtx
    
    def GetCamDistorsion(self):
        return self.cam.dist

    def LoadMarker(self,markerPath):
        self.marker.img=cv.imread(markerPath,cv.IMREAD_COLOR)
        self.marker.kp, self.marker.des = self.descriptor.detectAndCompute(cv.cvtColor(self.marker.img, cv.COLOR_BGR2GRAY), None)
        print('[INFO] Marker used is \"'+str(markerPath)+'\"')
    
    def GetFrame(self):
        # Capture frame-by-frame
        if self.cam.capture.isOpened() :
            ret, frame = self.cam.capture.read()

            if self.cam.video !=0 :
                # Video from file
                if not(self.cam.realMode==True) :
                    for i in range(int(self.cam.capture.get(cv.CAP_PROP_FPS))) :
                        if  cv.waitKey(1) & 0xFF == ord(str(self.escapeKey)):
                            # Stop with key
                            self.cam.capture.release()
                            cv.destroyAllWindows()
                            quit()

                if not ret :
                    if self.cam.loop==True :
                        self.cam.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
                        return self.GetFrame()
                    else :
                        # End of video file
                        self.cam.capture.release()
                        cv.destroyAllWindows()
                        quit()
                else:
                    # resize video frame
                    frame = cv.resize(frame,(self.cam.width,self.cam.height))
            else :
                # Video from camera check return
                if not ret :
                    print('[ERROR] Unable to capture video')
                    self.cam.capture.release()
                    cv.destroyAllWindows()
                    quit()    

            if  cv.waitKey(1) & 0xFF == ord(str(self.escapeKey)):
                # Stop with key
                self.cam.capture.release()
                cv.destroyAllWindows()
                quit()
        else :
            print('[ERROR] Unable to capture video source')
            self.cam.capture.release()
            cv.destroyAllWindows()
            quit()

        return frame
    
    def ComputeMatches(self,frame):
        good_matches = []
        k=2
        frame_kp, frame_des = self.descriptor.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)

        if frame_des is not None and len(frame_kp)>=k:
            matches=self.matcher.knnMatch(self.marker.des,frame_des,k=k)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])
        else:
            Log('[Warning] Not enough descriptors found in the frame')

        return good_matches,frame_kp

    def ComputeHomography(self,matches,frame_kp,minMatches=10):
        homography=None
        mask=None
        
        if len(matches) >minMatches :      
            src_points = np.float32([self.marker.kp[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_points = np.float32([frame_kp[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)
            homography, mask=cv.findHomography(src_points,dst_points,cv.RANSAC,ransacReprojThreshold=3.0)

        return homography,mask
    
    def RefineMatches(self,matches,frame_kp):
        correct_matches=[]
        homography,mask=self.ComputeHomography(matches,frame_kp)
        
        if homography is not None :
            # Get inliers mask
            correct_matches = [matches[i] for i in range(len(matches)) if mask[i]]

        return correct_matches
    
    def FindMarker(self,frame,homography,minMatches=10):
        found=False
        #@TODO
        #if homography is not None :
            # Warp marker image using homography
            # warped=cv.warpPerspective(src=cv.cvtColor(self.marker, cv.COLOR_BGR2GRAY), M=homography,dsize=(frame.shape[0],frame.shape[1]),flags=cv.WARP_INVERSE_MAP | cv.INTER_CUBIC)  
            # #matches,wrapped_kp_=self.ComputeMatches(warped,minMatches)
            # cv.imshow('test',warped)
        return found

    def ComputePose(self,frame,homography):
        maxSize = max(frame.shape[1],frame.shape[0]) #normalize dimension
        rvecs=None
        tvecs=None
        if homography is not None :
            axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
            
            points2D = np.float32([[0, 0], [frame.shape[1],0], [frame.shape[1],frame.shape[0]], [0, frame.shape[0]]]).reshape(-1, 1, 2)
            points3D = np.float32([[-frame.shape[1]/maxSize,-frame.shape[0]/maxSize,0],[frame.shape[1]/maxSize,-frame.shape[0]/maxSize,0],[frame.shape[1]/maxSize,frame.shape[0]/maxSize,0],[-frame.shape[1]/maxSize,frame.shape[0]/maxSize,0]])
            
            corners2D=cv.perspectiveTransform(points2D, homography)
            
            # Find the rotation and translation vectors.
            _,rvecs, tvecs = cv.solvePnP(points3D, corners2D, self.cam.mtx, self.cam.dist)
        else :
            Log('[Warning] Cannot compute pose without homography')  

        return rvecs, tvecs
