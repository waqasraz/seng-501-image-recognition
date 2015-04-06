package org.seng.image_recognition.tools;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.seng.image_recognition.utils.ImageAnalysis;
import org.seng.image_recognition.utils.ImageIO;

import static org.kohsuke.args4j.ExampleMode.ALL;

/**
 * Tool that performs the second stage of the image recognition process. Generates keypoints for each image.
 *
 * PERFORMED AS A MAPREDUCE TASK!
 *
 * Input:
 *      - file containing path/type pairs for each image
 * Ouput:
 *      - file containing path/type/keypoint pairs for each image
 *          img1Path car kp1 kp2 ...
 *          img2Path car kp1 kp2 ...
 *          img3Path bicycle kp1 kp2 ...
 *          img4Path bicycle kp1 kp2 ...
 *          img5Path motorbike kp1 kp2 ...
 *          img6Path motorbike kp1 kp2 ...
 *          ....
 */
public class KPExtractorTool extends HadoopTool {
    public static String TRAIN_DIR_PATH_KEY = "train_dir";
    public static String MAX_KEYPOINTS_KEY = "max_keypoints";

    /**
     * Command line options parser for kpextractor
     */
    public static class KPExtractorToolOptions {
        private String[] args;

        @Option(name = "-dir", usage = "path to training directory", metaVar = "TRAINING_DIR", required = true)
        private String trainDirPath;

        @Option(name = "-map", usage = "path to analyzeResults map file", metaVar = "TRAIN_MAP_FILE", required = true)
        private String mapFilePath;

        @Option(name = "-max", usage = "max # keypoints")
        private Integer maxKeypoints = 10000;

        @Option(name = "-o", usage = "path to save the hadoop results", metaVar = "OUTPUT_DIR", required = true)
        private String outputPath;

        public KPExtractorToolOptions(String[] args) {
            this.args = args;
        }

        public String getTrainDirPath() {
            return this.trainDirPath;
        }

        public String getMapFilePath() {
            return this.mapFilePath;
        }

        public String getOutputPath() {
            return this.outputPath;
        }

        public Integer getMaxKeypoints() {
            return this.maxKeypoints;
        }

        /**
         * Prepare the options
         */
        public void prepare() {
            final CmdLineParser parser = new CmdLineParser(this);
            try {
                parser.parseArgument(this.args);
            } catch (final CmdLineException e) {
                System.err.println(e.getMessage());
                System.err.println("Example: java kpextractor " + parser.printExample(ALL));
                parser.printUsage(System.err);
                System.err.println();

                System.exit(1);
            }
        }
    }

    /**
     * Mapper class for kpextractor
     */
    public static class KPExtractorMapper extends Mapper<Text, Text, Text, Text> {
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            //Load configuration
            StringTokenizer itr = new StringTokenizer(key.toString(), " ");
            String relPath = itr.nextToken();
            String trainDirPath = context.getConfiguration().get(TRAIN_DIR_PATH_KEY);
            Integer maxKeypoints = Integer.parseInt(context.getConfiguration().get(MAX_KEYPOINTS_KEY));

            //Load image from hdfs
            InputStream file = ImageIO.streamFromHDFS("hdfs://" + trainDirPath + relPath, context.getConfiguration());
            FImage image = ImageUtilities.readF(file);

            //Extract keypoints from image
            AbstractDenseSIFT<FImage> sift = ImageAnalysis.getSift();
            sift.analyseImage(image);
            LocalFeatureList<ByteDSIFTKeypoint> keypoints = sift.getByteKeypoints(0.005f);

            //Limit the number of keypoints, according to input settings
            if (keypoints.size() > maxKeypoints)
                keypoints = keypoints.subList(0, maxKeypoints);

            ByteArrayOutputStream baos = new ByteArrayOutputStream();

            //Write keypoints to output stream
            PrintWriter writer = new PrintWriter(baos);
            ImageIO.writeASCII(keypoints, writer);
            writer.close();

            //Write output stream as mapper output
            ImageIO.writeToContext(context, key, baos);
            baos.close();
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        //Read command line options
        final KPExtractorToolOptions options = new KPExtractorToolOptions(args);
        options.prepare();

        //Configure settings sent to mapper function
        Configuration conf = this.getConf();
        conf.set(TRAIN_DIR_PATH_KEY, options.getTrainDirPath());
        conf.set(MAX_KEYPOINTS_KEY, options.getMaxKeypoints().toString());

        //Prepare job
        Job job = new Job(conf, "kpextractor");
        job.setJarByClass(KPExtractorTool.class);
        job.setMapperClass(KPExtractorMapper.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(options.getMapFilePath()));
        FileOutputFormat.setOutputPath(job, new Path(options.getOutputPath()));

        //Run mapreduce job
        return job.waitForCompletion(true) ? 0 : 1;
    }

    /**
     * Runs the kpextractor tool
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new KPExtractorTool(), args);
    }
}