package org.seng.image_classification;

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.openimaj.hadoop.sequencefile.TextBytesSequenceFileUtility;

/**
 * Simple class to read the contents of a Hadoop sequence file
 */
public class SequenceFileReader {
	public static final String SEQUENCE_FILE_PATH = "../part1/HDFS_OUTPUT/QuantisedSift/part-r-00000";
	
	String path;
	
	public SequenceFileReader(String path) {
		this.path = path;
	}
    
    public TextBytesSequenceFileUtility read() throws IOException {
        return new TextBytesSequenceFileUtility(this.path, true);
    }
    
    /**
	 * Quick demo on how to read the sequence file
	 */
    public static void main( String[] args ) throws IOException {
        SequenceFileReader reader = new SequenceFileReader(SequenceFileReader.SEQUENCE_FILE_PATH);
        
        for (Map.Entry<Text, BytesWritable> item : reader.read()) {
        	System.out.println(item.getValue());
        }
    }
}


