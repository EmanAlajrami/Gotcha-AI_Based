import Tkinter
from Tkinter import *
import urllib
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk


app =Tkinter.Tk()
app.title("home")
app.geometry('600x300+350+200')
#############################

########################################
labelText = StringVar()
labelText.set("out of the state of imaging click   (q)"+"\n"+"help  (h)")
labl1 = Label(app, textvariable=labelText, height=4)
labl1.pack()


################################### 

def face():


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outface.avi',fourcc, 20.0, (640,480))

    cap = cv2.VideoCapture(0)
    cap.set(3, 640) #WIDTH
    cap.set(4, 480) #HEIGHT

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    while(True):


        ret, frame = cap.read()

    # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    #print(len(faces))
        ff=len(faces)
   # print ff
        if ff==1:
            out.write(frame)
            print 'yes'
        else:
            print 'no'
        
    
        
    # Display the resulting frame
        for (x,y,w,h) in faces:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
             roi_gray = gray[y:y+h, x:x+w]
             roi_color = frame[y:y+h, x:x+w]
             eyes = eye_cascade.detectMultiScale(roi_gray)
             for (ex,ey,ew,eh) in eyes:
                 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        cv2.imshow('framea',frame)
   
  

    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

##################################################################end beep

def body():

    def inside(r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh
#############################
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outbody.avi',fourcc, 20.0, (640,480))
#############################

    def draw_detections(frame, rects, thickness = 5):
        for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(frame, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), thickness)


    if __name__ == '__main__':

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
        cap=cv2.VideoCapture(0)
        cap.set(3, 640) #WIDTH
        cap.set(4, 480) #HEIGHT
############################################
  
   


    while True:
################################ came web
      #  url='http://192.168.43.1:8080/shot.jpg'
       # imgResp=urllib.urlopen(url)
       # imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
       # img=cv2.imdecode(imgNp,-1)
##############################################
        _,frame=cap.read()
        found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        #print(len(w))
        ff=len(w)
        if ff>0:
            print 'yes'
            out.write(frame)
        else:
            print 'no'

        draw_detections(frame,found)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



###########################################################################################################################end beep2

button1 = Button(app, text="Start Camera face", width=20 , command = face)
button1.pack(side='bottom',padx=15,pady=15)

#######################################################################################################################
button2 = Button(app, text="Start Camera body", width=20 , command = body)
button2.pack(side='bottom',padx=15,pady=15)

####################################################################






# Load an image using OpenCV
cv_img = cv2.imread("back.jpg")
 
# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, no_channels = cv_img.shape
canvas = Tkinter.Canvas(app, width = width, height = height)
canvas.pack()

photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
canvas.create_image(0,0, image=photo, anchor=Tkinter.NW)


# Run the window loop





















app.mainloop() #display windows 






