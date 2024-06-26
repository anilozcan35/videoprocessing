# -*- coding: utf-8 -*-
"""get_shoplifting_data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1107RZE38DrLbSgCCmrXsdzKaIhM3MRdH
"""

!wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/r3yjf35hzr-1.zip

!unzip r3yjf35hzr-1.zip -d ./

!mv 'Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad' shopliftingdata

!unzip /content/shopliftingdata/Dataset.zip -d ./

!cd /content/Dataset

!pip install -U kora

import cv2
from google.colab.patches import cv2_imshow
import time
from IPython import display

cap = cv2.VideoCapture('/content/Dataset/Shoplifting/Shoplifting (10).mp4')

while cap.isOpened():
#while True:
  ok, frame = cap.read()

  if not ok:
    break

  if ok:
    #edit your video size here, to adjust the performance
    largura=frame.shape[1]
    altura=frame.shape[0]
    lamenor=int(frame.shape[1]/2)
    altmenor=int(frame.shape[0]/2)
    frame = cv2.resize(frame, (lamenor,altmenor))

    # as you read
    display.clear_output(wait=True)
    cv2_imshow(frame)
    #delay time to update frame
    time.sleep(0.5)

  if cv2.waitKey(1100) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()