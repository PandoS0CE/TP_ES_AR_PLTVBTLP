import numpy as np
import cv2 as cv
import argparse

def get_args(calibration,marker):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--calibration", help="calibration file")
    parser.add_argument("-m","--marker", help="marker image")
    args = parser.parse_args()

    if args.calibration != None :
        calibration=str(args.calibration)
    if args.marker != None :
        marker=str(args.marker)

def capture(cam,quitKey):
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:
        print('Unable to capture video')
        cam.release()
        cv.destroyAllWindows()
        quit()  

    if  cv.waitKey(1) & 0xFF == ord(str(quitKey)):
        cam.release()
        cv.destroyAllWindows()
        quit()
    
    return frame

def main():
    #region Initialization
    cam = cv.VideoCapture(0,cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    calibration_file='./calibration/huawei_p30/calibration.npz'
    marker_file='./markers/fiducial.png'
    quitKey='q'
    get_args(calibration_file,marker_file)
    # Load calibration saved
    with np.load(str(calibration_file)) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    print('[INFO] Calibration file used is \"'+str(calibration_file)+'\"')
    print('[INFO] Marker used is \"'+str(marker_file)+'\"')
    print('[INFO] Press \"'+str(quitKey)+'\" to quit')
    #endregion
    
    #region Descriptor features
    orb = cv.ORB_create()
    # Find the keypoints and compute descriptor with ORB
    marker=cv.imread(marker_file,cv.IMREAD_COLOR)
    #marker = cv.cvtColor(marker,cv.COLOR_BGR2GRAY)
    kp_marker, des_marker = orb.detectAndCompute(marker, None) 

    marker_keypoints=cv.drawKeypoints(marker,kp_marker,_,color=(0,255,0), flags=0)
    size=(int(cam.get(cv.CAP_PROP_FRAME_WIDTH)),int(cam.get(cv.CAP_PROP_FRAME_HEIGHT)))
    marker_keypoints = cv.resize(marker_keypoints,size)
    cv.imshow('Marker', marker_keypoints)
    #endregion

    while(True):        
        # Process the frame
        frame=capture(cam,quitKey)
        cv.imshow('Camera', frame)
        
        # Pose Estimation
        #@TODO use mtx & co + papier prof   cv find homography     




if __name__ == '__main__':
    main()