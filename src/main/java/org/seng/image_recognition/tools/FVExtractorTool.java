package org.seng.image_recognition.tools;

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
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import org.seng.image_recognition.core.data.CentroidsData;
import org.seng.image_recognition.core.data.LocalCentroidsData;
import org.seng.image_recognition.utils.ImageAnalysis;
import org.seng.image_recognition.utils.ImageIO;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.StringTokenizer;

import static org.kohsuke.args4j.ExampleMode.ALL;

/**
 * Tool that performs the fourth stage of the image recognition process. Generates feature vectors for each training
 * image using the hard assigner found in stage 3.
 *
 * PERFORMED AS A MAPREDUCE TASK!
 *
 * Input:
 *      - file containing path/type pairs for each image
 *      - centroids file generated in TrainerTool
 * Ouput:
 *      - file containing path/type/feature-point pairs for each image
 *          img1Path car fp1 fp1 ...
 *          img2Path car fp1 fp2 ...
 *          img3Path bicycle fp1 fp2 ...
 *          img4Path bicycle fp1 fp2 ...
 *          img5Path motorbike fp1 fp2 ...
 *          img6Path motorbike fp1 fp2 ...
 *          ....
 */
public class FVExtractorTool extends HadoopTool {
    public static String TRAIN_DIR_PATH_KEY = "train_dir";
    public static String CENTROIDS_PATH = "centroids_path";

    /**
     * Command line options parser for fvextractor
     */
    public static class FVExtractorToolOptions {
        private String[] args;

        @Option(name = "-dir", usage = "path to training directory", metaVar = "TRAINING_DIR", required = true)
        private String trainDir;

        @Option(name = "-map", usage = "path to analyzeResults map file", metaVar = "TRAIN_MAP_FILE", required = true)
        private String mapFilePath;

        @Option(name = "-cen", usage = "path to centroids file", metaVar = "CENTROIDS_FILE", required = true)
        private String centroidsPath;

        @Option(name = "-o", usage = "path to save the hadoop results", metaVar = "OUTPUT_DIR", required = true)
        private String outputPath;

        public FVExtractorToolOptions(String[] args) {
            this.args = args;
        }

        public String getTrainDir() {
            return this.trainDir;
        }

        public String getMapFilePath() {
            return this.mapFilePath;
        }

        public String getCentroidsPath() {
            return this.centroidsPath;
        }

        public String getOutputPath() {
            return this.outputPath;
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
                System.err.println("Example: java fvextractor " + parser.printExample(ALL));
                parser.printUsage(System.err);
                System.err.println();

                System.exit(1);
            }
        }
    }

    /**
     * Mapper class for fvextractor
     */
    public static class FVExtractorMapper extends Mapper<Text, Text, Text, Text> {
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            //Load configuration
            StringTokenizer itr = new StringTokenizer(key.toString(), " ");
            String relPath = itr.nextToken();
            Configuration conf = context.getConfiguration();
            String trainDirPath = conf.get(TRAIN_DIR_PATH_KEY);
            String centroidsPath = conf.get(CENTROIDS_PATH);

            //Load image from hdfs
            InputStream file = ImageIO.streamFromHDFS("hdfs://" + trainDirPath + relPath, conf);
            FImage image = ImageUtilities.readF(file);

            //Read assigner from hdfs
            InputStream centroidsFile = ImageIO.streamFromHDFS("hdfs://" + centroidsPath, conf);
            CentroidsData centroidsData = LocalCentroidsData.read(centroidsFile);
            HardAssigner<byte[], float[], IntFloatPair> assigner = centroidsData.getCentroids().defaultHardAssigner();

            DoubleFV fv = ImageAnalysis.extractFeatures(assigner, image);

            ByteArrayOutputStream baos = new ByteArrayOutputStream();

            //Write keypoints to output stream
            PrintWriter writer = new PrintWriter(baos);
            ImageIO.writeASCII(fv, writer);
            writer.close();

            //Write output stream as mapper output
            ImageIO.writeToContext(context, key, baos);
            baos.close();
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        //Read command line options
        final FVExtractorToolOptions options = new FVExtractorToolOptions(args);
        options.prepare();

        //Configure settings sent to mapper function
        Configuration conf = this.getConf();
        conf.set(TRAIN_DIR_PATH_KEY, options.getTrainDir());
        conf.set(CENTROIDS_PATH, options.getCentroidsPath());

        //Prepare job
        Job job = new Job(conf, "fvextractor");
        job.setJarByClass(FVExtractorTool.class);
        job.setMapperClass(FVExtractorMapper.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(options.getMapFilePath()));
        FileOutputFormat.setOutputPath(job, new Path(options.getOutputPath()));

        //Run mapreduce job
        return job.waitForCompletion(true) ? 0 : 1;
    }

    /**
     * Runs the fvextractor tool
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new FVExtractorTool(), args);
    }
}