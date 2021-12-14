import React from "react";
import styled from "styled-components";
import Image from "assets/images/capture.PNG";

const Container = styled.div`
  display: flex;
  margin: 15px 0;
`;

const VideoImage = styled.img`
  width: 168px;
  height: 94px;
  margin: 0 10px 0 0;
`;

const VideoContent = styled.div`
  .title {
    font-weight: 800;
    font-size: 14px;
    margin-bottom: 5px;
  }
  .description {
    font-size: 12px;
  }
`;

function VideoListItem() {
  return (
    <Container>
      <VideoImage src={Image} />
      <VideoContent>
        <div className="title">2021년 11월 7일 15:30</div>
        <div className="description">수원시 영통구</div>
      </VideoContent>
    </Container>
  );
}

export default VideoListItem;
