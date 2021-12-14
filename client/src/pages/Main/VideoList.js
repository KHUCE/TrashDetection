import React, { useState, useEffect } from "react";
import styled from "styled-components";

import VideoListItem from "pages/Main/VideoListItem";

import { getVideoList } from "apis";
import Download from "assets/images/download.png";

const Container = styled.div`
  border-left: 1px solid #bbb;
  padding: 0 40px;
  overflow: auto;
  .title {
    font-size: 20px;
    margin-bottom: 20px;
  }
  .item {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    display: flex;
  }
  .description {
    margin-left: 10px;
    font-size: 14px;
    word-break: break-all;
    position: relative;
  }
`;

function VideoList() {
  const [data, setData] = useState([]);

  useEffect(() => {
    async function fetchVideo() {
      const _data = await getVideoList();
      setData(_data.data.VideoUrlList);
    }
    fetchVideo();
  }, []);

  useEffect(() => {
    console.log("fetched data", data);
  }, [data]);

  return (
    <Container>
      <div className="title">무단투기 검출 목록</div>
      {data.map((value) => {
        return (
          <div className="item">
            <video controls width="220">
              <source src={value} type="video/mp4" />
            </video>
            <div className="description">
              <div style={{ marginBottom: "10px" }}>수원시 영통구 영통동</div>
              <div style={{ fontSize: "12px" }}>
                {`발생일시 : ${value.substr(68, 4)}년 ${value.substr(
                  72,
                  2
                )}월 ${value.substr(74, 2)}일 ${value.substr(
                  76,
                  2
                )}시 ${value.substr(78, 2)}분 ${value.substr(80, 2)}초`}
              </div>
              <a
                href={value}
                download
                style={{ position: "absolute", bottom: "0", right: "5px" }}
              >
                <img src={Download} alt="다운로드버튼" width="15px" />
              </a>
            </div>
            {/* <a href={value} download></a>  */}
          </div>
        );
      })}
    </Container>
  );
}

export default VideoList;
