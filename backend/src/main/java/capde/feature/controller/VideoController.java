package capde.feature.controller;

import capde.core.utils.TCorpMap;
import capde.feature.service.VideoService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;

@RestController
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
@RequestMapping("/video")
public class VideoController {

    private final VideoService videoService;

    @GetMapping("/download")
    public TCorpMap getVideos() throws IOException{
        return videoService.getVideos();
    }

    @GetMapping("/txt_read")
    public String read() throws IOException {
        videoService.readObject("dumpedLog.txt");
        return "txt_read";
    }
}
