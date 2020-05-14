from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import threading
import time
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

global videoName
videoName = 1
detectVideoName = 1
totalDown = 0


def videoDetection():
    global detectVideoName
    global totalDown
    print("detect video" + str(detectVideoName))
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    # MobileNet sınıf etiketleri listesi
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # model yükleme
    print("[INFO] loading model...")
    net = cv2.dnn.readNet("mobilenet_ssd/MobileNetSSD_deploy.caffemodel", "mobilenet_ssd/MobileNetSSD_deploy.prototxt")

    print("[INFO] opening video file...")
    vs = cv2.VideoCapture("videos/" + str(detectVideoName) + ".avi")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # kare boyutlarını başlat
    # (videodaki ilk kareyi okuduktan hemen sonra bunları ayarlayacağız)
    W = None
    H = None
    roi = 250

    # centroid tracker'ımızı başlattıktan sonra, dlib korelasyon izleyicilerimizin
    # her birini depolamak için bir liste başlatın, ardından her bir benzersiz
    # nesne kimliğini bir TrackableObject öğesine eşlemek için bir dictinory ekleyin.
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # Şimdiye kadar işlenen toplam kare sayısını
    # ve aşağı veya yukarı hareket eden toplam nesne sayısını belirt
    totalFrames = 0


    # saniye başına işlem tahmincisi başına kare başlatmak
    fps = FPS().start()

    # video akışından karelerin döngüsü
    i = 0
    while True:

        # Bir sonraki kareyi alıp işleme
        frame = vs.read()
        frame = frame[1]

        # eğer bir video görüntülüyorsak ve bir kare almadıysak, o zaman videonun sonuna ulaştık
        if frame is None:
            detectVideoName += 1
            fps.stop()
            vs.release()
            cv2.destroyWindow("Frame")
            if int(detectVideoName) > 1:
                print("videos/" + str(detectVideoName - 1) + ".avi")
                os.remove("videos/" + str(detectVideoName - 1) + ".avi")

            videoDetection()


        # En fazla 500 piksel genişliğe sahip olacak şekilde çerçeveyi yeniden boyutlandırın
        # (ne kadar az veriye sahipsek, o kadar hızlı işleyebiliriz),
        # sonra çerçeveyi dlib için BGR'den RGB'ye dönüştürün
        frame = imutils.resize(frame, width=800)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # çerçeve boyutları boşsa, bunları ayarlayın.
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Mevcut durumu, (1) nesne algılayıcımız ya da
        # (2) korelasyon izci tarafından döndürülen sınırlama kutusu dikdörtgenleri listemizle birlikte başlatabilir.
        status = "Waiting"
        rects = []

        # algılama yöntemi kullanmamız gerekip gerekmediğini kontrol edin.
        if totalFrames % args["skip_frames"] == 0:
            # durumu ayarlayın ve yeni nesne izleyici setimizi başlatın
            status = "Detecting"
            trackers = []

            # çerçeveyi bir blob'a dönüştürün ve blob'u ağdan geçirin ve tespitleri alın
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # tespit edilenlerin döngüsü
            for i in np.arange(0, detections.shape[2]):

                # Tahmini ile ilgili olasılık çıkarma
                confidence = detections[0, 0, i, 2]

                # minimum güven gerektirerek zayıf tespitleri filtreleyin
                if confidence > args["confidence"]:

                    # sınıf etiketinin dizinini algılama listesinden çıkar
                    idx = int(detections[0, 0, i, 1])

                    # eğer sınıf etiketi bir insan değilse, yoksay
                    if CLASSES[idx] != "person":
                        continue

                    # Nesne için sınırlayıcı kutunun (x, y) koordinatlarını hesaplar.
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    # print("box: ",box.astype("int"))
                    (startX, startY, endX, endY) = box.astype("int")

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    # sınırlayıcı kutu koordinatlarından bir dlib dikdörtgen nesnesi oluşturun ve
                    # sonra dlib korelasyon izleyicisini başlatın
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    tracker.start_track(rgb, rect)

                    # tracker'ı tracker listemize ekleyin, böylece frameleri atlamak için kullanabiliriz.(?)
                    trackers.append(tracker)


        # Aksi takdirde, daha yüksek bir kare işleme verimi elde etmek için nesne dedektörlerimizden ziyade,
        # nesne izleyicilerimizi kullanmalıyız.
        else:
            for tracker in trackers:
                # sistemimizin durumunu “Tracking” olarak ayarlamak
                status = "Tracking"

                # izleyiciyi güncelle ve güncellenmiş konumu yakala
                tracker.update(rgb)
                pos = tracker.get_position()

                # konum nesnesini açmak
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # sınırlayıcı kutu koordinatlarını dikdörtgenler listesine ekleyin
                rects.append((startX, startY, endX, endY))

        # çerçevenin ortasına yatay bir çizgi çizin
        # bir nesne bu çizgiyi geçtikten sonra 'yukarı' mı 'aşağı' mı hareket ettiklerini belirleyeceğiz
        cv2.line(frame, (0, roi), (W, roi), (0, 255, 255), 2)

        # eski nesne centroidlerini yeni hesaplanan nesne centroidleriyle ilişkilendirmek için centroid izleyicisini kullanın
        objects = ct.update(rects)

        # takip edilen nesnenin döngüsü

        for (objectID, centroid) in objects.items():
            # Geçerli nesne kimliği için izlenebilir bir nesne olup olmadığını kontrol edin
            to = trackableObjects.get(objectID, None)

            # izlenebilir bir nesne yoksa, bir tane oluşturun
            if to is None:
                to = TrackableObject(objectID, centroid)

            # Aksi takdirde, izlenebilir bir nesne vardır,
            # böylece yönünü belirlemek için onu kullanabiliriz
            else:
                # şimdiki  centroid'in y koordinatı ile  önceki centroidlerin ortalaması
                # arasındaki fark bize nesnenin hangi yönde hareket ettiğini söyler
                # ('yukarı' için negatif ve 'aşağı' için pozitif)
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # nesnenin sayılıp sayılmadığını görmek için kontrol edin
                if not to.counted:

                    # yön pozitifse (nesnenin aşağı doğru hareket ettiğini gösterir)
                    # centroid orta çizginin altındaysa, nesneyi say.
                    # elif direction > 0 and centroid[1] > H // 2:
                    if direction > 0 and centroid[1] > roi:
                        print("box: ", box.astype("int"))
                        print("person")

                        frameNew = frame[startY: startY + endY, startX: startX + endX]
                        # cv2.imshow("frameNew", frameNew)

                        imagesFolder = "./images"

                        cv2.imwrite(imagesFolder + "/image_" + str(int(totalDown)) + ".jpg", frame)
                        i += 1

                        totalDown += 1
                        to.counted = True

            # zlenebilir nesneyi dictonyde saklayın
            trackableObjects[objectID] = to

            # Çıktı karesinde hem nesnenin kimliğini hem de nesnenin centroidini çizin
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # çerçevede göstereceğimiz bir bilgi grubu oluşturun
        info = [
            ("Down", totalDown),
            ("Status", status)
        ]

        # bunları çerçeve üzerine çizin
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # çıktı çerçevesini göster
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(10) & 0xFF

        # q ile çıkış
        if key == ord("q"):
            break

        # şu ana kadar işlenen toplam kare sayısını artırın ve
        # ardından FPS sayacını güncelleyin
        totalFrames += 1
        fps.update()

    # zamanlayıcıyı durdurmak ve FPS bilgilerini görüntülemek
    fps.stop()
    print("Toplam giren:  ", totalDown)

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    vs.release()

    cv2.destroyAllWindows()


def videoRecording():
    global videoName

    print("vid:" +str(videoName))
    cap = cv2.VideoCapture("rtsp://admin:tunca3806@192.168.2.100:554/chID=13&steamType=main")

    if(cap.isOpened() == False):
        print("unable to read camera")

    start_time = time.time()
    print("start: " + str(start_time))

    frameWidth = int(cap.get(3))
    frameHeight = int(cap.get(4))


    #out = cv2.VideoWriter("videos/"+str(videoName)+".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (frameWidth, frameHeight))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter("videos/"+str(videoName) + ".avi" , fourcc, 20.0, (frameWidth, frameHeight))
    while(True):
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            frame = imutils.resize(frame, width = 640, height = 480)

            cv2.imshow("frame", frame)

            # 5 dakikada bir video kaydetmek için
            if (int(time.time() - start_time) > 10):
                videoName += 1
                out.release()
                cap.release()
                cv2.destroyWindow("frame")
                videoRecording()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    t1 = threading.Thread(target=videoRecording)
    t1.start()
    time.sleep(20)
    t0 = threading.Thread(target=videoDetection)
    t0.start()