#SON HALİ - 2 Özellik 1 duygu durumu -alt satır
import cv2
import dlib
import os
import time

t1 = time.time()
# detector yükle
detector = dlib.get_frontal_face_detector()

# predictor yükle
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#image dataset içindeki klasörleri listeler
fileList = os.listdir("imageDataset")

# .arff  dosyasi oluştur.
f1 = open("26_points_Duzenlenmis.arff", "w")
f1.write(f"@RELATION facepoints" +"\n")

for i in range(0,26,1):
    f1.write(f"@ATTRIBUTE x_point{i}  NUMERIC" + "\n")
    f1.write(f"@ATTRIBUTE y_point{i}  NUMERIC" + "\n")
    f1.write("\n")
f1.write("@ATTRIBUTE class        {anger,disgust,fear,joy,neutral,sadness,surprise}"+"\n"+"\n")
f1.write("@DATA")
f1.write("\n")

for emotions in fileList:
    imageList = os.listdir(f"imageDataset/{emotions}")
    print(imageList)

    for image in os.listdir(f"imageDataset/{emotions}"):
        # resmi oku
        img = cv2.imread(f"imageDataset/{emotions}//{image}")

        # Görüntüyü gri tonlamaya dönüştür
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Yer işaretlerini bulmak için dedektörü kullan
        faces = detector(gray)
        print(image)
        for face in faces:
            x1 = face.left() # left point
            y1 = face.top() # top point
            x2 = face.right() # right point
            y2 = face.bottom() # bottom point

            # Yer işareti nesnesi oluştur
            landmarks = predictor(image=gray, box=face)

            # Tüm noktalar arasında dolaşın
            for n in range(17, 68 ,2):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                print(f" {x} {y}")

                f1.write(f"{x},{y},")

                # bir daire çiz
                cv2.circle(img=img, center=(x, y), radius=3, color=(0,255,255), thickness=-1)

            f1.write(f"{emotions}\n")

    f1.write("\n")

f1.close()
t2=time.time()
x = t2-t1

saniye  = int(x % 60)
dakika = int((x // 60) % 60)
saat = int((x // 60) // 60)
print(f".arff dosyasının oluşturulması {saat}:{dakika}:{saniye} sürdü")

# resmi göster
cv2.imshow(winname="Face", mat=img)

# Her kare arasındaki gecikme
cv2.waitKey(delay=0)

# Tüm pencereleri kapat
cv2.destroyAllWindows()

