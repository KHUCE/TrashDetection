import axios from "axios";

export async function getVideoList() {
  try {
    const data = await axios.get("http://18.141.6.10:8080/video/download");
    console.log("data:", data);
    return data;
  } catch {
    alert("data patch 오류 발생");
  }
}
