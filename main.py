import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
import ARTools
from ARTools import ARPipeline

def get_args(calibration,marker,minMatches,maxMatches,video):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--calibration', help='calibration filepath')
    parser.add_argument('-m','--marker', help='marker image path')
    parser.add_argument('-min','--minmatches', help='min matches')
    parser.add_argument('-max','--maxmatches', help='max matches')
    parser.add_argument('-v','--video',help='video path')
    parser.add_argument('-r','--real',help='play video at fps rate in real time',action='store_true')
    args = parser.parse_args()

    if args.calibration != None :
        calibration=str(args.calibration)
    if args.marker != None :
        marker=str(args.marker)
    if args.minmatches != None :
        minMatches=int(args.minmatches)
    if args.maxmatches != None :
        maxMatches=int(args.maxmatches)
    if args.video != None :
        video=str(args.video)
    
    return calibration,marker,minMatches,maxMatches,video,not(args.real)

def main():
    #region Initialization
    video=0 #0 to use camera
    calibration_file='./videos/genius_F100/calibration/calibration.npz'
    marker_file='./markers/fiducial.png'
    minMatches=10
    maxMatches=20
    calibration_file,marker_file,minMatches,maxMatches,video,realMode=get_args(calibration_file,marker_file,minMatches,maxMatches,video)
    moveWindows=True
    pipeline=ARPipeline(video=video,realMode=realMode)
    pipeline.LoadCamCalibration(calibration_file)
    pipeline.LoadMarker(marker_file)
    #endregion
    
    while(True): 
        frame=pipeline.GetFrame()
        matches,frame_kp=pipeline.ComputeMatches(frame=frame,minMatches=minMatches)
        pipeline.HomographyEstimation(matches,frame_kp)
        #pipeline.ComputePose()
        #region Rendering
        if matches is not None :
            cv.imshow('AR Camera',frame)
            cv.imshow('Keypoints',ARTools.DrawKeypoints(frame,frame_kp))
            img_matches=ARTools.DrawMatches(frame,frame_kp,pipeline.GetMarker(),pipeline.GetMarkerKeypoints(),matches,maxMatches=maxMatches)
            img_matches = cv.resize(img_matches,(frame.shape[1],frame.shape[0]))
            cv.imshow('Matches',img_matches)
        else :
            # not enough feature to compute pose
            cv.imshow('AR Camera', frame)
            cv.imshow('Keypoints', frame)
            img_matches=ARTools.DrawMatches(frame,frame_kp,pipeline.GetMarker(),pipeline.GetMarkerKeypoints(),None)
            img_matches = cv.resize(img_matches,(frame.shape[1],frame.shape[0]))
            cv.imshow('Matches',img_matches)
        
        if moveWindows==True :
            # init position windows once
            cv.moveWindow('AR Camera',0,0)
            cv.moveWindow('Keypoints',frame.shape[1],0)
            cv.moveWindow('Matches',2*frame.shape[1],0)
            moveWindows=False
        #endregion
if __name__ == '__main__':
    main()