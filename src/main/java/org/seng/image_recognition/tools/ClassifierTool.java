package org.seng.image_recognition.tools;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import org.seng.image_recognition.core.data.CentroidsData;
import org.seng.image_recognition.core.data.FVData;
import org.seng.image_recognition.core.data.LocalCentroidsData;
import org.seng.image_recognition.core.data.LocalFVData;
import org.seng.image_recognition.utils.ImageAnalysis;

import java.io.IOException;
import java.util.List;

import static org.kohsuke.args4j.ExampleMode.ALL;


/**
 * Tool that performs the final stage of the image recognition process. Uses the output data from the previous
 * stages to attempt to classify images given by the user
 *
 * PERFORMED ON LOCAL MACHINE
 */
public class ClassifierTool {

    /**
     * Command line options parser for classifier
     */
    public static class ClassifierToolOptions {
        private String[] args;

        @Option(name = "-fv", usage = "path to features file", metaVar = "KEYPOINTS_FILE", required = true)
        private String featuresPath;

        @Option(name = "-cen", usage = "path to centroids file", metaVar = "CENTROIDS_FILE", required = true)
        private String centroidsPath;

        public ClassifierToolOptions(String[] args) {
            this.args = args;
        }

        public String getFeaturesPath() {
            return this.featuresPath;
        }

        public String getCentroidsPath() {
            return this.centroidsPath;
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
    public static void main(String[] args) throws IOException {
        final ClassifierToolOptions options = new ClassifierToolOptions(args);
        options.prepare();

        List<FVData> features = LocalFVData.readList(options.getFeaturesPath());
        CentroidsData centroids = LocalCentroidsData.read(options.getCentroidsPath());
        HardAssigner<byte[], float[], IntFloatPair> assigner = centroids.getCentroids().defaultHardAssigner();

        LiblinearAnnotator<FVData, String> ann = ImageAnalysis.trainAnnotator(features);

        ClassifierToolGUI app = new ClassifierToolGUI(ann, assigner);
    }
}
