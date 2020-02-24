import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse
import ARTools
from ARTools import ARPipeline

def get_args(calibration,marker,minMatches,maxMatches,video):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--calibration', help='calibration file')
    parser.add_argument('-m','--marker', help='marker image')
    parser.add_argument('-min','--minmatches', help='min matches')
    parser.add_argument('-max','--maxmatches', help='max matches')
    parser.add_argument('-v','--video',help='video path')
    parser.add_argument('-f','--fast',help='play video in at fps time',action='store_true')
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
    
    return calibration,marker,minMatches,maxMatches,video,args.fast

def main():
    #region Initialization
    video=0 #0 to use camera
    calibration_file='./calibration/huawei_p30/calibration.npz'
    marker_file='./markers/fiducial.png'
    minMatches=10
    maxMatches=20
    calibration_file,marker_file,minMatches,maxMatches,video,fastMode=get_args(calibration_file,marker_file,minMatches,maxMatches,video)
    moveWindows=True
    pipeline=ARPipeline(video=video,fastMode=fastMode)
    pipeline.LoadCamCalibration(calibration_file)
    pipeline.LoadMarker(marker_file)
    #endregion
    
    while(True): 
        frame=pipeline.GetFrame()
        matches,frame_kp=pipeline.ComputeMatches(frame=frame,minMatches=minMatches)
        pipeline.HomographyEstimation(matches,frame_kp)
        #pipeline.ComputePose()
        #region rendering
        if matches is not None :
            cv.imshow('Camera',frame)
            cv.imshow('Keypoints',ARTools.DrawKeypoints(frame,frame_kp))
            img_matches=ARTools.DrawMatches(frame,frame_kp,pipeline.GetMarker(),pipeline.GetMarkerKeypoints(),matches,maxMatches=maxMatches)
            img_matches = cv.resize(img_matches,(frame.shape[1],frame.shape[0]))
            cv.imshow('Matches',img_matches)
        else :
            #not enough feature to compute pose
            cv.imshow('Camera', frame)
            cv.imshow('Keypoints', frame)
            img_matches=ARTools.DrawMatches(frame,frame_kp,pipeline.GetMarker(),pipeline.GetMarkerKeypoints(),None)
            img_matches = cv.resize(img_matches,(frame.shape[1],frame.shape[0]))
            cv.imshow('Matches',img_matches)
        
        if moveWindows==True :
            cv.moveWindow('Camera',0,0)
            cv.moveWindow('Keypoints',frame.shape[1],0)
            cv.moveWindow('Matches',2*frame.shape[1],0)
            moveWindows=False
        #endregion
if __name__ == '__main__':
    main()