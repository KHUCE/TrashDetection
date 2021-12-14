import React from "react";
import styled from "styled-components";

const Container = styled.div`
  padding: 10px 40px;
  background-color: #fff;

  .headerText {
    font-size: 30px;
    font-weight: 800;
  }
`;

function Header() {
  return (
    <Container>
      <div className="headerText">쓰레기 무단투기 실시간 감시</div>
    </Container>
  );
}

export default Header;
