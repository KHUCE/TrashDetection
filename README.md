## 쓰레기 무단투기 실시간 감지 시스템
### 2021-2 캡스톤디자인 - 이영구 교수님 A조



## 🧑‍💻 팀원
### 컴퓨터공학과 김다희
### 컴퓨터공학과 이학주
### 컴퓨터공학과 황성연



## 🖥 소개
### yolov5와 deepsort를 사용하여 쓰레기 객체와 인물 객체를 인식하고, 객체 간의 겹침을 인식한 후, 분리됨을 감지하고 쓰레기의 좌표 고정 시간을 체크하여 쓰레기 투기로 인식하도록 하였습니다.


## 🏢 아키텍처
<img width="384" alt="스크린샷 2021-12-15 오전 12 41 00" src="https://user-images.githubusercontent.com/30286254/146030251-b0fb9965-ea86-4511-ad01-a1ecb55bc4d6.png">
rtsp를 통해 ip 카메라에서 송출된 영상을 실시간으로 스트리밍하고, 10초 단위로 동영상이 Amazon RDS로 저장됩니다. 쓰레기 투기가 감지되면 녹화 중인 동영상 파일 이름을 저장하여 감지된 리스트에 띄울 수 있도록 하였으며,
react client에서 실시간 영상과, 쓰레기 투기가 감지된 영상들을 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/30286254/146036567-e22d15a9-2ab2-4ac9-ba51-b7e6eb54e6d6.png)


## ⭐️ 사용 가이드

***
Server

    git clone https://github.com/KHUCE/TrashDetection
    
    cd TrashDetection/yolov5
    
    !pip install requirements.txt
    
    python run.py --source 0(rtsp 주소 보유시 rtsp 주소) --weights weights/trash.pt
 

***
Client

    cd ../client
    
    yarn
    
    yarn start
    
