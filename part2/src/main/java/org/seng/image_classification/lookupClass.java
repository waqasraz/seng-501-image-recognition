package org.seng.image_classification;

import org.apache.hadoop.io.BytesWritable;
import org.openimaj.hadoop.sequencefile.SequenceFileUtility;
import org.openimaj.hadoop.sequencefile.TextBytesSequenceFileUtility;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;

import org.apache.hadoop.io.Text;
/**
 * Created by waqas on 16/03/15.
 */
public class lookupClass {

    final String SEQUENCE_FILE_PATH = "/home/waqas/Data/train-map.txt";
    final static Charset ENCODING = StandardCharsets.UTF_8;
    public Map <String,String> myMap = new HashMap<String,String>();

    public lookupClass() throws IOException  {
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

   public String get_Class(String s){
        String get = myMap.get(s);

        if(get.contains("car")){
            return "car";
        } else if (get.contains("motorbike")){
            return "motorbike";
        }else if (get.contains("bicycle")){
            return "bicycle";
        }
        return "NA";
    }


}
