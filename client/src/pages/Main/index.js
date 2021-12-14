import React from "react";
import styled from "styled-components";

import VideoStream from "pages/Main/VideoStream";
import VideoList from "pages/Main/VideoList";

const Container = styled.div`
  display: flex;
  padding: 20px 0 100px;
  max-width: 100%;
  justify-content: space-between;
`;

function Main() {
  return (
    <Container>
      <VideoStream />
      <VideoList />
    </Container>
  );
}

export default Main;
