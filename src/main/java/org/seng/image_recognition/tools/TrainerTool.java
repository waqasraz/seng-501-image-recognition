package org.seng.image_recognition.tools;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.seng.image_recognition.core.data.KPData;
import org.seng.image_recognition.core.data.LocalCentroidsData;
import org.seng.image_recognition.core.data.LocalKPData;
import org.seng.image_recognition.utils.ImageAnalysis;

import java.util.List;

import static org.kohsuke.args4j.ExampleMode.ALL;

/**
 * Tool that performs the third stage of the image recognition process. Trains the quantizer based on the given keypoints.
 *
 * PERFORMED ON LOCAL MACHINE
 *
 * Input:
 *      - keypoints file generated in KPExtractorTool
 * Ouput:
 *      - file containing the training data centroids.
 */
public class TrainerTool {
    /**
     * Command line options parser for trainer
     */
    public static class TrainerToolOptions {
        private String[] args;

        @Option(name = "-kp", usage = "path to keypoints file", metaVar = "KEYPOINTS_FILE", required = true)
        private String keypointsPath;

        @Option(name = "-o", usage = "path to save the centroids file", metaVar = "OUTPUT_FILE", required = true)
        private String outputPath;

        public TrainerToolOptions(String[] args) {
            this.args = args;
        }

        public String getKeypointsPath() {
            return this.keypointsPath;
        }

        public String getOutputPath() {
            return this.outputPath;
        }

        public void prepare() {
            final CmdLineParser parser = new CmdLineParser(this);
            try {
                parser.parseArgument(this.args);
            } catch (final CmdLineException e) {
                System.err.println(e.getMessage());
                System.err.println("Example: java trainer " + parser.printExample(ALL));
                parser.printUsage(System.err);
                System.err.println();

                System.exit(1);
            }
        }
    }

    /**
     * Main function for performing the trainer task
     * @param keypointsPath path to the keypoints file
     * @param outputPath path to write centroids data to
     */
    public static void train(String keypointsPath, String outputPath) throws Exception {
        //Read keypoints
        List<KPData> keypoints = LocalKPData.readList(keypointsPath);

        //Train quantizer
        ByteCentroidsResult centroids = ImageAnalysis.trainQuantiser(keypoints);

        LocalCentroidsData data = new LocalCentroidsData(centroids);
        data.write(outputPath);
    }

    /**
     * Runs the trainer tool
     */
    public static void main(String[] args) throws Exception {
        final TrainerToolOptions options = new TrainerToolOptions(args);
        options.prepare();

        System.out.println("Training quantizer (this will take several minutes, maybe even a few hours...)");
        train(options.getKeypointsPath(), options.getOutputPath());
        System.out.println("Done!");
    }
}