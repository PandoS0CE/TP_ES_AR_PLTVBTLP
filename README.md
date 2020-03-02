# TP_ES_AR_PLTVBTLP







## Pipeline
1. Load cam calibration
2. Load marker
3. Build marker points and features
Loop :        
4. Get Frame
5. Find marker in frame
    * convert frame to gray
    * extract frame features 
    * get frame matches with marker (knn,hamming...)
        * get good matches apply ratio test for outliers, min treshold
    * refine matches with homography
        * compute homography + mask
        * apply mask to remove outlier
    * improve found homography
        * by warping marker on frame with homography previously compute
        * extract features (kp,des) of wraped image
        * get matches + refine + homography
6. Compute pose 
7. Rendering
8. Tunning


optim : 
+ 3.training
+ 7.tracking old frames




Liens utiles : 

https://docs.opencv.org/master/d9/dab/tutorial_homography.html#tutorial_homography_Demo1


https://books.google.fr/books?id=kYdgCgAAQBAJ&pg=PA176&lpg=PA176&dq=perspectiveTransform+opencv+homography+augmented&source=bl&ots=TbHb8PdOUb&sig=ACfU3U0fl1W--c6HqaU7fTzZo6GTA1pBOg&hl=fr&sa=X&ved=2ahUKEwja2PSLwPvnAhUcSxUIHSwdDFMQ6AEwA3oECAoQAQ#v=onepage&q=perspectiveTransform%20opencv%20homography%20augmented&f=false

AR en python avec opencv, sift et homography
https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/


Un code src avec un petit cpp https://github.com/quantyle/augmented-reality-opencv
sa video :  https://www.youtube.com/watch?v=yNvmfDJDXFM

Un mec qui explique vite zef et qui donne un tuto au debut 
https://stackoverflow.com/questions/12283675/augmented-reality-sdk-with-opencv
le tuto en question :  https://dsynflo.blogspot.com/2010/06/simplar-augmented-reality-for-opencv.html

Une version matlab qui marche bien
https://www.youtube.com/watch?v=JIh_rE1IcQc
mais le code de la page existe plus

Un mec qui utilise opencv for unity
https://github.com/MasteringOpenCV/code
https://www.youtube.com/watch?v=B4pc_e8mdcs


## Conda

[OpenCV](https://anaconda.org/conda-forge/opencv)
`conda install -c conda-forge opencv`

[MatPlotLib](https://anaconda.org/conda-forge/matplotlib)
`conda install -c conda-forge matplotlib`

[OpenGL](https://anaconda.org/conda-forge/pyopengl)
`conda install -c conda-forge pyopengl`

[OpenGL Accelerate](https://anaconda.org/anaconda/pyopengl-accelerate)
`conda install -c anaconda pyopengl-accelerate`