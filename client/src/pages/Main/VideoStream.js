import React, { useState, useEffect } from "react";
import styled from "styled-components";
import moment from "moment";
import ReactPlayer from "react-player";
import video from "assets/images/testvideo.mp4";

const Container = styled.div`
  padding: 0 40px;
  .description {
    margin: 10px 0;
    font-size: 24px;
  }
`;

const VideoArea = styled.div`
  background-color: lightgray;
  width: 64vw;
  height: 36vw;
`;

function VideoStream() {
  let timer = null;
  const [time, setTime] = useState(moment()); //useState 훅을 통해 time 값 디폴트 설정

  useEffect(() => {
    timer = setInterval(() => {
      //timer 라는 변수에 인터벌 종료를 위해 저장
      setTime(moment()); // 현재 시간 세팅
    }, 1000); //1000ms = 1s 간 반복
    return () => {
      clearInterval(timer); // 함수 언마운트시 clearInterval
    };
  }, []);

  return (
    <Container>
      <ReactPlayer
        url="rtsp://18.141.6.10:554"
        playing={true}
        width={960}
        height={540}
      />
      <div className="description">
        수원시 영통구 영통동 992-10 {time.format("YYYY-MM-DD HH:mm:ss")}
      </div>
    </Container>
  );
}

export default VideoStream;
