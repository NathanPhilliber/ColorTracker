import cv2
import numpy
import math
import random
import time
import os

vidCap = cv2.VideoCapture(0)
keepGoing = True #Used for outer while loop to switch between select and track
drawing = False #Used for custom selection box draw
ix,iy = 0, 0 #Used for point 1 on selection
nx, ny = 0, 0 #Used to store second point of selection
disCor = False #Used to turn of reset box in upper left corner
showDebug = False #Used to display FPS 
imgCleanCus = None #Used to get custom selection without drawings on it

n = 150 #Used for reset box size X
m = 100 #Used for reset box size Y
roi = [] #Used to make histogram
hsv_roi = [] #Used to make histogram
xS, xB = [], [] #Used to get custom selection mouse coords
yS, yB = [], [] #Used to get custom selection mouse coords
roi_hist = [] #Histogram for camshift tracking
mask = [] #Mask for camshift tracking

swap = [] #Used for menu variables, booleans

height, width, depth = 1,1,1 #Define hwd, set to arbitrary values

xF = [] #Used to keep track of 
yF = []
hsv = []
dst = []
ret2 = []

lastFTime = 1
frameCount = 0
fps = 0

videoImport = cv2.VideoCapture("input.avi")

font = cv2.FONT_HERSHEY_SIMPLEX
track_window = []
img = None
imgClean = []
doneSel = False
trackPointsList = []
colorPointsList = []
trackPoints = []
colorPoints = []
maxPoints = 50
makeNewSel = True
selCount = 0
selLimit = 5
goBackToSelection = False
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) #limit iterations 
itera = 0 

numOfPics = 16
uniqueOptions = 8

filename = None
madeVideo = False

#============================================================================#
#adds image 1 under image 2- big = second                                    
def stitchImages(img1, img2): 
    global numOfPics
    
    (hgt1, wid1, dep1)= img1.shape
    (hgt2, wid2, dep2)= img2.shape
    
    newHgt = hgt1 + hgt2
    newImage = numpy.zeros((newHgt, wid1, 3), numpy.uint8)
    
    newImage[0:hgt2, 0:wid2] = img2
    newImage[hgt2:hgt1+hgt2, 0:wid1] = img1
    
    return (newImage)
#============================================================================#
#Builds and displays the menu
def showMenu():
    global swap, uniqueOptions
    
    img = getOption(0, swap[0])
    
    for i in range(uniqueOptions - 1):
        img = stitchImages(getOption(i + 1, swap[i + 1]), img)  
        
    cv2.imshow("Menu", img)
#============================================================================# 
#returns the apppropriate image button
def getOption(num,swap):
    global numOfPics
    
    picList = []
    
    for i in range(numOfPics):
        picList.append(cv2.imread("MenuButtons/button_" + str(i) + ".png"))  
        
    if num == 0:
        if swap:
            img = picList[8] 
        else:
            img = picList[9] 
    elif num == 1:
        if swap:
            img = picList[13] 
        else:
            img = picList[12]     
    elif num == 2:
        if swap:
            img = picList[6] 
        else:
            img = picList[5]    
    elif num == 3:
        if swap:
            img = picList[3] 
        else:
            img = picList[4]     
    elif num == 4:
        if swap:
            img = picList[1] 
        else:
            img = picList[12]     
    elif num == 5:
        if swap:
            img = picList[0] 
        else:
            img = picList[10]   
    elif num == 6:
        if swap:
            img = picList[7] 
        else:
            img = picList[11] 
    elif num == 7:
        if swap:
            img = picList[15] 
        else:
            img = picList[14]             
    else:
        img = picList[num]
        
    return img 
#============================================================================#
#Called everytime the menu is clicked
def menuButton(event, x, y, flags, param): 
    global swap, madeVideo
    
    height = 75
    width = 250
    
    if event == cv2.EVENT_LBUTTONDOWN:
        button = int(y/75)
        swap[button] = not swap[button]
        
        if swap[6] == False:
            madeVideo = True
        
        showMenu()
#============================================================================#
#Used to swap out with the mouse listener, does nothing
def do_nothing(event,x,y,flags,param):
    
    zero = 0
#============================================================================#
#Used for custom selection 
def draw_rect(event, x, y, flags, param): 
    global ix, iy, drawing, nx, ny
    
    if drawing == True:
        cv2.rectangle(img, (ix, iy), (x,y), (20, 20, 255), 2) 
        
    if event == cv2.EVENT_LBUTTONDOWN:
         
        drawing = True
        ix, iy = x, y 
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            nx, ny = x, y
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix,iy),(x,y),(20, 20, 255), 2)
#============================================================================#
#Used for custom selection, draws box even when mouse stops moving
def draw_rect_no_event(img): 
    global ix, iy, drawing, nx, ny, imageCleanCus
    
    if drawing == True:
        cv2.rectangle(img, (ix,iy),(nx,ny),(20, 20, 255), 2)
        return numpy.copy(imgCleanCus)
    
    return img  
#============================================================================#
#Takes image and draws reset box on it
def boxMask(frame): 
    global disCor, xF, yF, m, n, keepGoing, goBackToSelection, height, width
    
    if swap[5] == False:
        
        (bc,gc,rc) = cv2.split(frame)
        list = [rc, bc, gc]
        new = cv2.merge(list)
        mask = numpy.zeros((height, width, 1), numpy.uint8)
        mask[:m, :n] = 255
        newMask = cv2.bitwise_and(new, frame, mask = mask)
        mask2 = 255-mask
        img5 = cv2.add(frame, 0, mask = mask2)
        frame = img5 + newMask
        cv2.rectangle(frame, (n,m), (0,0), (100,100,200), 4)

        for i in range(len(xF)):
            if xF[i] < n and yF[i] < m:
                goBackToSelection = True
                break 
        
    return frame
#============================================================================# 
#Calculates the fps
def calculateFPS(): 
    global fps, frameCount, lastFTime
    
    time1 = time.time()
    frameCount += 1
    
    if time1-lastFTime >= 1:
        lastFTime = time1
        fps = frameCount
        frameCount = 0
#============================================================================#  
#First half of program, makes the selections
def makeSelections(): 
    global img, vidCap, doneSel, ix, iy, nx, ny, keepGoing, makeNewSel, font, selCount, track_window, imgCleanCus, drawing, xS, Xb, yS, yB, imgCleanIn, itera, roi_hist, hsv_roi, mask, roi, goBackToSelection, imgClean, swap, numOfPics, videoImport, madeVideo
    
    makeNewSel = True
    selCount = 0
    drawing = False
    ix, iy, yS, yB, xS, xB = [],[],[],[],[],[]
    itera = 0
    roi_hist = []
    mask = []   
    roi = []
    hsv_roi = []    
    imgClean = []
    img = None
    lastFTime = 1
    frameCount = 0
    fps = 0   
    doneSel = False
    track_window = []
    madeVideo = False
    
    swap = []
    
    for i in range(numOfPics):
        swap.append(True)    

    if vidCap.isOpened() or videoImport.isOpened():
        goBackToSelection = False
        
        if swap[2] == False: #Check of reading from webcam or file
            ret, img = videoImport.read() #####################
            if ret != True:
                videoImport = cv2.VideoCapture("input.avi")
                ret, img = videoImport.read()
            
        else:
            ret, img = vidCap.read()
            
        height, width, depth = img.shape             
        x, y, w, h = width/4, height/8, (width/4*3)-width/4, (height/8*7)-height/8 #Default selection size 
        
        showMenu()
        cv2.setMouseCallback("Menu", menuButton)
        
        while makeNewSel: #Loop per selection
            
            if swap[2] == False: #Check of reading from webcam or file
                ret, img = videoImport.read()
                if ret != True:
                    videoImport = cv2.VideoCapture("input.avi")
                    ret, img = videoImport.read()                
            else:
                ret, img = vidCap.read()
                
            doneSel = False
            
            x, y, w, h = width/4, height/8, (width/4*3)-width/4, (height/8*7)-height/8 #Default selection size
            ix, iy, nx, ny = 0,0,0,0  
            
            while keepGoing:  
                height, width, depth = img.shape             
                x, y, w, h = width/4, height/8, (width/4*3)-width/4, (height/8*7)-height/8 #Default selection size                 
                
                if swap[2] == False: #Check of reading from webcam or file
                    ret, img = videoImport.read()
                    if ret != True:
                        videoImport = cv2.VideoCapture("input.avi")
                        ret, img = videoImport.read()                    
                else:
                    ret, img = vidCap.read()
                    
                cv2.rectangle(img, (x, y), (w+x, h+y), (20, 20, 255), 3)
                
                cv2.putText(img, "Selection #" + str(selCount + 1), (10,50), font, 1, (80, 80, 255), 2)
                
                cv2.imshow("Color Tracker", img)
                
                k = cv2.waitKey(30) & 0xff
                
                if k == 13 or swap[1] == False: #enter
                    
                    if swap[2] == False: #Check of reading from webcam or file
                        ret, imgCleanIn = videoImport.read()
                        if ret != True:
                            videoImport = cv2.VideoCapture("input.avi")
                            ret, imgCleanIn = videoImport.read()                
                    else:
                        ret, imgCleanIn = vidCap.read()                    
                    
                    track_window.append((x, y, w, h)) 
                    
                    swap[1] = True
                    showMenu()
                    break
                
                if swap[0] == False:
                    
                    makeNewSel = False
                    
                    break
                       
                if k == 27 or swap[7] == False: #esc
                    
                    keepGoing = False
                    ret, imgCleanIn = vidCap.read()
                    
                    break 
                
                if k == 49 or swap[4] == False: #Bring up custom draw screen
                    
                    cv2.setMouseCallback("Color Tracker", draw_rect) #Add click events
                    
                    if swap[2] == False: #Check of reading from webcam or file
                        ret, imgCleanCus = videoImport.read()
                        if ret != True:
                            videoImport = cv2.VideoCapture("input.avi")
                            ret, imgCleanCus = videoImport.read()                        
                    else:
                        ret, imgCleanCus = vidCap.read()  
                        
                    if swap[2] == False: #Check of reading from webcam or file
                        ret, imgCleanIn = videoImport.read()
                        if ret != True:
                            videoImport = cv2.VideoCapture("input.avi")
                            ret, imgCleanIn = videoImport.read()                
                    else:
                        ret, imgCleanIn = vidCap.read()  
                        
                    img = numpy.copy(imgCleanCus)
                    
                    while doneSel != True:
                        
                        img = draw_rect_no_event(img)
                        
                        k = cv2.waitKey(30) & 0xFF
                        
                        cv2.imshow("Color Tracker",img)
                        
                        if k == 13 or swap[1] == False: #On pressing enter
                            cv2.setMouseCallback("Color Tracker", do_nothing) #Stop listening to mouse 
                            x, y, w, h = ix, iy, nx-ix, ny-iy
                            
                            track_window.append((ix, iy, abs(nx-ix), abs(ny-iy)))
                            doneSel = True
                            drawing = False
                            swap[1] = True
                            swap[4] = True
                            showMenu()
                            break
                        
                        if swap[0] == False:
                            doneSel = True
                            drawing = False
                            swap[4] = True
                            cv2.setMouseCallback("Color Tracker", do_nothing) #Stop listening to mouse 
                            break  
                        
                        if k == 52:
                            makeNewSel = not makeNewSel
                        elif k == 27 or swap[7] == False: #On pressing esc
                            keepGoing = False
                            cv2.setMouseCallback("Color Tracker", do_nothing)     
                            break
                        
                    if doneSel == True:
                        break
                    
                if doneSel == True and makeNewSel == False:
                    break
                
            if swap[0] == False:
                break
            
            selCount += 1
            if selCount >= selLimit:
                makeNewSel = False
                swap[1] = False
                swap[0] = False
                showMenu()
            
                            
            #Same as img, except will not have anything drawn on it        
            imgClean.append(imgCleanIn)      
        
            xS.append(min(x, x+w))
            xB.append(max(x, x+w))  #Clean image dimensions so there are no negatives
            yS.append(min(y, y+h))
            yB.append(max(y, y+h)) 
                         
            roi.append(imgCleanIn[(yS[-1]):(yB[-1]), (xS[-1]):(xB[-1])]) #Create cropped image from selection  
            
            #cv2.imshow("ROI_" + str(itera),roi[-1]) #Display
                        
            hsv_roi.append(cv2.cvtColor(roi[-1], cv2.COLOR_BGR2HSV))  #Convert roi to hsv
            
            #cv2.imshow("HSVROI_" + str(itera),hsv_roi[-1]) #Display
                       
            mask.append(cv2.inRange(hsv_roi[-1], numpy.array((0., 60.,32.)), numpy.array((180.,255.,255.))))  #Create mask
            
            #cv2.imshow("MASK_" + str(itera),mask[-1]) #Display
                      
            roi_hist.append(cv2.calcHist([hsv_roi[-1]],[0],mask[-1],[180],[0,180])) #Create histogram
            cv2.normalize(roi_hist[-1],roi_hist[-1],0,255,cv2.NORM_MINMAX) #Normalize histogram
            itera += 1        
        
    else:
        print "No Camera Connected"   
#============================================================================# 
#Second half of program, tracks objects
def startTracking(): 
    global vidCap, keepGoing, disCor, trackPointsList, colorsPointsList, fps, track_window, showDebug, term_crit, goBackToSelection, roi_hist, height, width, depth, xF, yF, hist, dst, ret2, videoImport, filename
    
    if swap[2] == False: #Check of reading from webcam or file
        ret, frame = videoImport.read() 
        if ret != True:
            videoImport = cv2.VideoCapture("input.avi")
            ret, frame = videoImport.read()          
    else:
        ret, frame = vidCap.read() 
        
    trackPointsList = []
    colorPointsList = []
    xF = []
    yF = []
    hsv = []
    dst = []
    ret2 = []
    
    filename = "Track_" + str(random.randrange(10000000)) + ".avi"
    
    wid = vidCap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    hgt = vidCap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) 
           
    size = (int(wid), int(hgt))
    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    writ = cv2.VideoWriter()
    writ.open(filename, fourcc, 25.0, size, 1)   
    
    #START TRACKING
    if vidCap.isOpened():
        if swap[2] == False: #Check of reading from webcam or file
            ret, frame = videoImport.read() 
            if ret != True:
                videoImport = cv2.VideoCapture("input.avi")
                ret, frame = videoImport.read()
                    
        else:
            ret, frame = vidCap.read()        
        
        (height, width, depth) = frame.shape

        while keepGoing:  
            
            if track_window != None:
                
                for track in track_window:
                
                    xF.append(track[0]) 
                    yF.append(track[1])
                    ret2.append(None)
                    hsv.append(None)
                    dst.append(None)
                    trackPointsList.append(None)
                    colorPointsList.append(None)
 
            if swap[2] == False: #Check of reading from webcam or file
                ret, frame = videoImport.read() 
                if ret != True:
                    videoImport = cv2.VideoCapture("input.avi")
                    ret, frame = videoImport.read()
                                
            else:
                ret, frame = vidCap.read()             
            
            
            if swap[5] == False:
                frame = boxMask(frame)
            
            if showDebug or swap[3] == False:
                cv2.putText(frame, "FPS: " + str(fps), (10,60), font, 1, (20, 20, 255), 2)
            
            for i in range(len(track_window)):
                hsv[i] = (cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))  #Convert entire image to hsv
                #h, s, v = cv2.split(hsv[i])
                #cv2.imshow("HSV_1", h)
                
                dst[i] = (cv2.calcBackProject([hsv[i]],[0],roi_hist[i],[0,180],1))  #Clc back project for entire frame and histogram made previously
                #cv2.imshow("DST_1", dst[i])
                
                if track_window[i] < (5,5,5,5):
                    track_window[i] = (1000,1000,1000,1000)  
                    
                if track_window[i] == (0,0,0,0): #Make sure track_window is not 0
                    track_window[i] = (1,1,1,1)
                
                ret2[i], track_window[i] = cv2.CamShift(dst[i], track_window[i], term_crit) #Use camshift to adjust tracking window
                
                pts = cv2.cv.BoxPoints(ret2[i]) #Draw tracking box
                pts = numpy.int0(pts)
                img2 = cv2.polylines(frame,[pts],True, 255,2) 
                
            cv2.imshow("Color Tracker", frame) #Display
            
            
            calculateFPS()
            
            k = cv2.waitKey(1)
            
            if k == 32 or swap[0] == True: #Wait for spacebar, exit tracking loop
                goBackToSelection = True
                writ.release()
                
            if k == 27 or swap[7] == False: #Exit both while loops on esc
                keepGoing = False
                writ.release()
                break
            
            if k == 50:
                disCor = not disCor
                
            if k == 51:
                showDebug = not showDebug
            if goBackToSelection:
                keepGoing = False
                break
            
            if swap[1] == False:
                swap[1] = True
                showMenu()
                
            if swap[4] == False:
                swap[4] = True
                showMenu()
                
            if swap[5] == False and swap[2] == False:
                swap[5] = True
                showMenu()
            
            if swap[6] == False:
                writ.write(frame)
       
    else:
        print "No Camera Connected"
        
    if goBackToSelection:
        
        keepGoing = True  
        
##############################################################################
#______________________________End of Functions______________________________#
##############################################################################

#Main loop
while keepGoing:
    
    makeSelections()
    
    startTracking()
    
    if madeVideo == False:
        print filename
        #os.remove(filename)
    
cv2.destroyAllWindows()
vidCap.release()