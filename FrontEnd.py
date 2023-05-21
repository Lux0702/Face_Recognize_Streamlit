import streamlit as st
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import cv2 as cv
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import io
from streamlit_option_menu import option_menu
import fontawesome as fa
import subprocess
import joblib
import Chapter3 as c3
import argparse
import os
from firebase_admin import credentials, firestore,storage
from firebase_admin import db
import firebase_admin
        
global imgout
FRAME_WINDOW = st.image([])


def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

if 'add_selectbox' not in st.session_state:
    st.session_state.add_selectbox = True

if 'Li_thuyet_XLA_selected' not in st.session_state:
    st.session_state.Li_thuyet_XLA_selected = True
with st.sidebar:
    add_selectbox = option_menu("Main Menu", ["Home", 'Get_Data_Face','Training','Face_Recognize','Li thuyet XLA'], 
        icons=['house' ,'task' ,'gear','search'], menu_icon="cast", default_index=0)
    #add_selectbox
if 'key' not in st.session_state:
    st.session_state.key = 0
if add_selectbox=="Home":
  st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>NHẬN DẠNG KHUÔN MẶT</h1>", unsafe_allow_html=True)
  st.image("https://huviron.com.vn/FileUpload/Images/1.jpg")
  st.text('''Thành viên nhóm thực hiện:
                  1️⃣ Nguyễn Thanh Sang   20110710
                  2️⃣ Lê Anh Kiệt         20110684
            Đề tài : Nhận dạng chữ số bằng thuật toán KNN và CNN
            Link source code: github
            Link app đã deploy: www.''')
elif add_selectbox=="Get_Data_Face":
  st.session_state.add_selectbox = True
  st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>NHẬN DẠNG KHUÔN MẶT</h1>", unsafe_allow_html=True)
  Get_Data_Face = option_menu(
        menu_title = 'Get Data_Face',
        options = ['Data'],
        orientation = "horizontal",
        menu_icon='house',
        icons=['pencil']
        #display_toobar="reset"
    )


  if Get_Data_Face=='Data':
    st.header("Get Data User via Face - ✏️")
    st.write('Nhập tên không dấu viết liền.Lưu ý: Viết hoa ở chữ cái đầu của mỗi từ')
    input=st.text_input(label='Nhập văn bản', value='')
    FRAME_WINDOW.empty()
    if(st.button('Enter')):
        path=os.path.abspath('./image/'+input)
        FRAME_WINDOW = st.image([])
        while os.path.exists(path):
            st.write("Name Already Taken")
            input=st.text_input(label='Nhập văn bản', value='')
            
        os.makedirs(path)
        cap = cv.VideoCapture(0)
        parser = argparse.ArgumentParser()
        parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
        parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
        parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
        parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
        parser.add_argument('--face_detection_model', '-fd', type=str, default='./model/face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
        parser.add_argument('--face_recognition_model', '-fr', type=str, default='./model/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
        parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
        parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
        parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
        parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
        args = parser.parse_args()
        def visualize(input, faces, fps, thickness=2):
            if faces[1] is not None:
                for idx, face in enumerate(faces[1]):
                    print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                    coords = face[:-1].astype(np.int32)
                    cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                    cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                    cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                    cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                    cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                    cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
            cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if __name__ == '__main__':
            detector = cv.FaceDetectorYN.create(
          'face_detection_yunet_2022mar.onnx',
          "",
          (320, 320),
          0.9,
          0.3,
          5000)
      
            recognizer = cv.FaceRecognizerSF.create(
      'face_recognition_sface_2021dec.onnx',"")
            
            tm = cv.TickMeter()

            frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            detector.setInputSize([frameWidth, frameHeight])
            imgBackground = cv.imread('./background.png')
            dem = 0
            while (dem<300 and add_selectbox=="Get_Data_Face"):
                hasFrame, frame = cap.read()
                if not hasFrame:
                    print('No frames grabbed!')
                    break

                # Inference
                tm.start()
                faces = detector.detect(frame) # faces is a tuple
                tm.stop()
                
                key = cv.waitKey(1)
                if key == 27:
                    break

                if (True):
                    if faces[1] is not None:
                        face_align = recognizer.alignCrop(frame, faces[1][0])
                        file_name = path +'/'+input+'_%04d.bmp' % dem
                        cv.imwrite(file_name, face_align)
                        dem = dem + 1
                # Draw results on the input image
                visualize(frame, faces, tm.getFPS())

                # Visualize results
                #imgBackground[162:162+480,55:55+640] = frame
                FRAME_WINDOW.image(frame, channels='BGR')
                #cv.imshow("Face Attedance", imgBackground)
                #cv.imshow('Live', frame)
            cv.destroyAllWindows()
            if(dem==300):
                st.success("Lấy dữ liệu khuôn mặt thành công.")
                st.info('Vui lòng qua tab training để cập nhật dữ liệu nhận diện mới')
                cap.release()
                FRAME_WINDOW.empty()
            
elif add_selectbox=="Face_Recognize":
  st.session_state.add_selectbox = False
  FRAME_WINDOW = st.image([])
  st.markdown("<style>#MainMenu { visibility: hidden; }</style>", unsafe_allow_html=True)
  cap = cv.VideoCapture(0)

  if 'stop' not in st.session_state:
      st.session_state.stop = False
      stop = False

  press = st.button('Stop')
  if press:
      if st.session_state.stop == False:
          st.session_state.stop = True
          st.button("Start again")
          cap.release()
      else:
          st.session_state.stop = False
          

  print('Trang thai nhan Stop', st.session_state.stop)

  if 'frame_stop' not in st.session_state:
      frame_stop = cv.imread('stop.jpg')
      st.session_state.frame_stop = frame_stop
      print('Đã load stop.jpg')

  if st.session_state.stop == True:
      FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


  svc = joblib.load('svc.pkl')
  with os.scandir("./image") as entries:
    mydict = [os.path.basename(entry.path) for entry in entries if entry.is_dir()]

  def visualize(input, faces, fps, thickness=2):
      if faces[1] is not None:
          for idx, face in enumerate(faces[1]):
              #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

              coords = face[:-1].astype(np.int32)
              cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
              cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
              cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
              cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
              cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
              cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
      cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      cv.putText(input,result,(coords[0], coords[1]),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


  if __name__ == '__main__':
      detector = cv.FaceDetectorYN.create(
          'face_detection_yunet_2022mar.onnx',
          "",
          (320, 320),
          0.9,
          0.3,
          5000)
      
      recognizer = cv.FaceRecognizerSF.create(
      'face_recognition_sface_2021dec.onnx',"")

      tm = cv.TickMeter()

      frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
      frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
      detector.setInputSize([frameWidth, frameHeight])

      dem = 0
      while add_selectbox=="Face_Recognize":
          
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            cv.putText(frame,result,(1,50),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
      cv.destroyAllWindows()  
elif add_selectbox=="Li thuyet XLA":
  placeholder = st.empty()
  st.session_state.add_selectbox = False
  st.markdown("<style>#MainMenu { visibility: hidden; }</style>", unsafe_allow_html=True)
  with st.sidebar:
    Li_thuyet_XLA_selected =  option_menu(
        menu_title='Xử lí ảnh',
        options=['Negative', 'Logarit', 'Power', 'PiecewiseLinear', 'Histogram', 'HistogramEqualization', 'LocalHistogram', 'HistogramStatistics', 'Smoothing', 'SmoothingGauss', 'MedianFilter', 'Sharpen', 'UnSharpMasking', 'Gradient'],
        orientation="vertical",
        menu_icon='image'
        #icons=['pencil']
    )  
  if 'Li_thuyet_XLA_selected' not in st.session_state or Li_thuyet_XLA_selected != st.session_state.Li_thuyet_XLA_selected:
    # Đặt lại ứng dụng về trạng thái ban đầu
    st.session_state.selected_option = Li_thuyet_XLA_selected
    if(Li_thuyet_XLA_selected=="Negative"):
        placeholder.empty()

        st.session_state.Li_thuyet_XLA_selected = False
        st.markdown("<style>#MainMenu { visibility: hidden; }</style>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
            btn_pre=st.button("OnNegative")
            if btn_pre:

                imgin = np.array(Image.open("digit.jpg").convert('L'))
                M, N = imgin.shape
                imgout = np.zeros((M, N), np.uint8)
                c3.Negative(imgin,imgout)
                st.image(imgout, caption='ImageOut')

    elif(Li_thuyet_XLA_selected=="Logarit"):
        st.session_state.Li_thuyet_XLA_selected = Li_thuyet_XLA_selected

        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnLogarit")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.Logarit(imgin,imgout)
            st.image(imgout, caption='ImageOut') 
    elif(Li_thuyet_XLA_selected=="Power"):
        st.session_state.Li_thuyet_XLA_selected = Li_thuyet_XLA_selected
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnPower")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.Power(imgin,imgout)
            st.image(imgout, caption='ImageOut')
    elif(Li_thuyet_XLA_selected=="PiecewiseLinear"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnPiecewiseLinear")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.PiecewiseLinear(imgin,imgout)
            st.image(imgout, caption='ImageOut')      
    elif(Li_thuyet_XLA_selected=="Histogram"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnHistogram")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.Logarit(imgin,imgout)
            st.image(imgout, caption='ImageOut')
    elif(Li_thuyet_XLA_selected=="HistogramEqualization"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnHistogramEqualization")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.HistogramEqualization(imgin,imgout)
            st.image(imgout, caption='ImageOut')  
    elif(Li_thuyet_XLA_selected=="LocalHistogram"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnLocalHistogram")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.LocalHistogram(imgin,imgout)
            st.image(imgout, caption='ImageOut')      
    elif(Li_thuyet_XLA_selected=="HistogramStatistics"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnHistogramStatistics")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.HistogramStatistics(imgin,imgout)
            st.image(imgout, caption='ImageOut')
    elif(Li_thuyet_XLA_selected=="Smoothing"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnSmoothing")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.Smoothing(imgin,imgout)
            st.image(imgout, caption='ImageOut')        
    elif(Li_thuyet_XLA_selected=="SmoothingGauss"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnSmoothingGauss")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.SmoothingGauss(imgin,imgout)
            st.image(imgout, caption='ImageOut')
    elif(Li_thuyet_XLA_selected=="MedianFilter"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnMedianFilter")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.MedianFilter(imgin,imgout)
            st.image(imgout, caption='ImageOut')
    elif(Li_thuyet_XLA_selected=="Sharpen"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnSharpen")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.Sharpen(imgin,imgout)
            st.image(imgout, caption='ImageOut')
    elif(Li_thuyet_XLA_selected=="UnSharpMasking"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("OnUnSharpMasking")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.UnSharpMasking(imgin,imgout)
            st.image(imgout, caption='ImageOut')
    elif(Li_thuyet_XLA_selected=="UnSharpMasking"):
        st.session_state.Li_thuyet_XLA_selected = False
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image)
            image.convert('RGB')
            image.save('digit.jpg')
        btn_pre=st.button("UnSharpMasking")
        if btn_pre:

            imgin = np.array(Image.open("digit.jpg").convert('L'))
            M, N = imgin.shape
            imgout = np.zeros((M, N), np.uint8)
            c3.UnSharpMasking(imgin,imgout)
            st.image(imgout, caption='ImageOut')
elif add_selectbox=="Training":
    st.session_state.Li_thuyet_XLA_selected=False
    st.header("Training data user via Face - 📌")
    st.write('Vui lòng nhấn nút Training và chờ trong giây lác để cập nhật dữ liệu nhận diện')
    if(st.button("Training")):
        #subprocess.run(["python", 'Training.py'], stdout=subprocess.PIPE, text=True)
        result = subprocess.run(['python', 'Training.py'], capture_output=True, text=True)
        if result.returncode == 0:
            st.success('Training thành công! Bạn có thể truy cập tab Face_Recognize để thử nhận diện khuôn mặt')
        else:
            st.error('Training thất bại! Hãy thử lại')







