package capde.feature.service;

import capde.core.utils.TCorpMap;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.GetObjectRequest;
import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectInputStream;
import com.amazonaws.util.IOUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import javax.transaction.Transactional;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class VideoService {

    @Value("${aws.s3.bucket}")
    private String bucket;

    private final AmazonS3 amazonS3;

    public List<String> readObject(String storedFileName) throws IOException{
        S3Object object = amazonS3.getObject(new GetObjectRequest(bucket, storedFileName));
        S3ObjectInputStream ois = null;
        BufferedReader br = null;

        List<String> videoList = new ArrayList<String>();

        String[] data = null;

        // Read the CSV one line at a time and process it.
        try {
            ois = object.getObjectContent();
            System.out.println ("ois = " + ois);
            br = new BufferedReader (new InputStreamReader(ois, "UTF-8"));
            String line;
            while ((line = br.readLine()) != null) {
                // Store 1 record in an array separated by commas
                data = line.split(",", 0);

                for (String s : data) {
                    videoList.add(s);
                    System.out.print(s);
                }
                System.out.println();

            }
        }finally {
            if(ois != null){
                ois.close();
            }
            if(br != null){
                br.close();
            }
        }

        return videoList;
    }

    @Transactional(rollbackOn = Exception.class)
    public TCorpMap getVideos() throws IOException{

        TCorpMap result = new TCorpMap();

        List<String> urlList = new ArrayList<String>();

        List<String> videoList = readObject("dumpedLog.txt");

        System.out.println(videoList.size());

        for(String s : videoList) {
            System.out.println(s);
            urlList.add("https://capstonedesignbucket.s3.ap-northeast-2.amazonaws.com/" + s);
        }

        result.put("VideoUrlList", urlList);
        result.put("rsltCd", "00");
        result.put("rsltMsg", "Success");

        return result;
    }
}
