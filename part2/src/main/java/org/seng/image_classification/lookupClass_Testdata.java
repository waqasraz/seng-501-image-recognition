package org.seng.image_classification;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by waqas on 16/03/15.
 */
public class lookupClass_Testdata {

    final String SEQUENCE_FILE_PATH = "/home/waqas/Data/test-map.txt";
    final static Charset ENCODING = StandardCharsets.UTF_8;
    public Map <String,String> myMap = new HashMap<String,String>();

    public lookupClass_Testdata() throws IOException  {
        Path path = Paths.get(SEQUENCE_FILE_PATH);
        Scanner scanner =  new Scanner(path, ENCODING.name());

        while (scanner.hasNextLine()){
            //process each line in some way
            String line = scanner.nextLine();
            String[] parts = line.split(" ");
            String key = parts[0];
            String value = parts[1];
            myMap.put(key, value);
        }

    }

   public String get_Image_Name(String key){
        String path = myMap.get(key);
        File name = new File(path);
       return name.getName();
    }


}
