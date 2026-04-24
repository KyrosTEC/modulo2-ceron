#Author: AECL 

import cv2
import numpy as np
import threading
import time

import grpc
import te3002b_pb2
import te3002b_pb2_grpc
import google.protobuf.empty_pb2

from actividad_2_04 import CenterLineDetector
from actividad_2_05 import TrafficLightDetection
from actividad_2_06 import TrafficSignDetection


class SimRobotNode():
    def __init__(self):
        self._addr="127.0.0.1"
        self.channel=grpc.insecure_channel(self._addr+':7072')
        self.stub = te3002b_pb2_grpc.TE3002BSimStub(self.channel)

        self.cv_image = None 
        self.result=None
        self.datacmd=te3002b_pb2.CommandData();
        self.dataconfig=te3002b_pb2.ConfigurationData();
        self.twist=[0,0,0,0,0,0]
        self.running = True
        self.timer_delta=0.025
        self.running_time=0.0

        self.detector = CenterLineDetector()
        self.traffic_light_detector = TrafficLightDetection()
        self.traffic_sign_detector = TrafficSignDetection()
        
    def callback(self):
        
        self.dataconfig.resetRobot = True
        self.dataconfig.mode = 2
        self.dataconfig.cameraWidth = 360
        self.dataconfig.cameraHeight = 240
        self.dataconfig.resetCamera = False
        self.dataconfig.scene = 2026
        self.dataconfig.cameraLinear.x = 0
        self.dataconfig.cameraLinear.y = 0
        self.dataconfig.cameraLinear.z = 0
        self.dataconfig.cameraAngular.x  = 0
        self.dataconfig.cameraAngular.y  = 0
        self.dataconfig.cameraAngular.z  = 0
        req=google.protobuf.empty_pb2.Empty()
        self.twist=[0.0,0.0,0.0,0.0,0.0,0.0]
        res=self.stub.SetConfiguration(self.dataconfig);
        self.dataconfig.resetRobot = False
        time.sleep(0.25)
        res=self.stub.SetConfiguration(self.dataconfig);

        while self.running:
            print("Time:" + str(self.running_time))
            self.result = self.stub.GetImageFrame(req)
            print("frame ID: " + str(self.result.seq))
            img_buffer=np.frombuffer(self.result.data, np.uint8)
            img_in=cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            img = img_in
            if img_in is not None:
                img=self.add_noise_to_image(img_in, 3)

            new_width = 320
            new_height = 240
            new_dim = (new_width, new_height)
            self.cv_image = cv2.resize(img, new_dim, interpolation=cv2.INTER_LANCZOS4)
            
            point = self.detector.detect_center_line(self.cv_image)

            if point is not None:
                cv2.circle(self.cv_image, point, 5, (0,255,0), -1)

            # Detectar estado del semáforo
            traffic_light_state = self.traffic_light_detector.detect_state(self.cv_image)
            print("Traffic Light State:", traffic_light_state)

            # Detectar señales de tráfico
            traffic_sign_label, traffic_sign_psnr, traffic_sign_bbox = self.traffic_sign_detector.detect_sign(self.cv_image)
            print("Traffic Sign Detected:", traffic_sign_label)

            # --- BLOQUE AÑADIDO: Dibujar cuadrado verde y etiqueta ---
            if traffic_sign_label != "none" and traffic_sign_bbox is not None:
                p1, p2 = traffic_sign_bbox
                cv2.rectangle(self.cv_image, p1, p2, (0, 255, 0), 2)
                cv2.putText(self.cv_image, f"{traffic_sign_label} ({traffic_sign_psnr:.1f})", 
                            (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # ---------------------------------------------------------

            # Movimiento robot (no lo tocamos)
            self.twist=[0.0,0.0,0.0,0.0,0.0,0.0]
            self.datacmd.linear.x=self.twist[0]
            self.datacmd.linear.y=self.twist[1]
            self.datacmd.linear.z=self.twist[2]
            self.datacmd.angular.x=self.twist[3]
            self.datacmd.angular.y=self.twist[4]
            self.datacmd.angular.z=self.twist[5]
            self.result=self.stub.SetCommand(self.datacmd);

            # Mostrar imagen
            cv2.imshow('Synthetic Image', self.cv_image)
            cv2.waitKey(1)

            time.sleep(self.timer_delta-0.001)
            self.running_time=self.running_time+self.timer_delta

    def add_noise_to_image(self,image, kernel_s, noise_level=5):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        noise = np.random.randint(-noise_level, noise_level + 1, v.shape, dtype='int16')
        v_noisy = v.astype('int16') + noise
        v_noisy = np.clip(v_noisy, 0, 255).astype('uint8')
        hsv_noisy = cv2.merge([h, s, v_noisy])
        noisy_image = cv2.cvtColor(hsv_noisy, cv2.COLOR_HSV2BGR)
        noisy_image = cv2.GaussianBlur(noisy_image, (kernel_s, kernel_s), 0)

        alpha = 0.55
        beta = 55
        output_image = cv2.convertScaleAbs(noisy_image, alpha=alpha, beta=beta)

        return output_image

def main(args=None):
    robot_node = SimRobotNode()
    thread = threading.Thread(target=robot_node.callback)
    thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping the thread...")
        robot_node.running = False
        thread.join()
        cv2.destroyAllWindows()
        print("Thread stopped")

if __name__ == "__main__":
    main()