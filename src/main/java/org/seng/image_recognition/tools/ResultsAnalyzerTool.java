package org.seng.image_recognition.tools;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import org.seng.image_recognition.core.data.*;
import org.seng.image_recognition.utils.BasicGroupedDataset;
import org.seng.image_recognition.utils.ImageAnalysis;

import java.util.List;
import java.util.Map;

import static org.kohsuke.args4j.ExampleMode.ALL;


/**
 * Optional stage of the image recognition process. Tests the accuracy of the image recognition process.
 *
 * Input:
 *      - centroids file generated in TrainerTool
 *      - feature vector file generated in FVExtractorTool
 *      - Training images
 * Output:
 *      - The accuracy of the image recognizer tool
 */
public class
        ResultsAnalyzerTool {
    /**
     * Command line options parser for analyzer
     */
    public static class ResultsAnalyzerToolOptions {
        private String[] args;

        @Option(name = "-cen", usage = "path to to centroids file", metaVar = "OUTPUT_FILE", required = true)
        private String centroidsPath;

        @Option(name = "-fv", usage = "path to feature vector file", metaVar = "FEATURE_VECTOR_FILE", required = true)
        private String featureVectorPath;

        @Option(name = "-dir", usage = "path to training directory", metaVar = "TRAINING_DIR", required = true)
        private String trainDirPath;

        @Option(name = "-n", usage = "# images to test per class")
        private Integer numTestImagesPerClass = 5;

        @Option(name = "-s", usage = "start test analysis on nth image in each class folder")
        private Integer start = 5;

        public ResultsAnalyzerToolOptions(String[] args) {
            this.args = args;
        }

        public String getFeatureVectorPath() {
            return this.featureVectorPath;
        }

        public String getCentroidsPath() {
            return this.centroidsPath;
        }

        public String getTrainDirPath() {
            return this.trainDirPath;
        }

        public int getNumTestImagesPerClass() {
            return this.numTestImagesPerClass;
        }

        public int getStart() {
            return this.start;
        }

        public void prepare() {
            final CmdLineParser parser = new CmdLineParser(this);
            try {
                parser.parseArgument(this.args);
            } catch (final CmdLineException e) {
                System.err.println(e.getMessage());
                System.err.println("Example: java analyzer " + parser.printExample(ALL));
                parser.printUsage(System.err);
                System.err.println();

                System.exit(1);
            }
        }
    }

    /**
     * Main function for performing the analyzer task
     * @param featureVectorPath path to feature vector file
     * @param centroidsPath path to centroids file
     * @param trainDirPath path to training folder
     * @param numTestImagesPerClass number of images to test per class
     * @param start start test analysis on nth image in each class folder
     */
    public static void analyzeResults(String featureVectorPath, String centroidsPath, String trainDirPath,
                                      int numTestImagesPerClass, int start) throws Exception {
        //Read features
        List<FVData> features = LocalFVData.readList(featureVectorPath);

        CentroidsData centroidsData = LocalCentroidsData.read(centroidsPath);
        HardAssigner<byte[], float[], IntFloatPair> assigner = centroidsData.getCentroids().defaultHardAssigner();

        //Train annotator
        System.out.println("Training annotator");
        LiblinearAnnotator<FVData, String> ann = ImageAnalysis.trainAnnotator(features);

        //Prepare test data
        BasicGroupedDataset<FVData> testDataset = ImageAnalysis.prepareTestData(
                trainDirPath, assigner, numTestImagesPerClass, start);

        //Test accuracy rate of annotator
        ClassificationEvaluator<CMResult<String>, String, FVData> eval =
                new ClassificationEvaluator<CMResult<String>, String, FVData>(
                        ann, testDataset, new CMAnalyser<FVData, String>(CMAnalyser.Strategy.SINGLE));
        Map<FVData, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);
    }

    /**
     * Runs the analyzer tool
     */
    public static void main(String[] args) throws Exception {
        final ResultsAnalyzerToolOptions options = new ResultsAnalyzerToolOptions(args);
        options.prepare();

        analyzeResults(options.getFeatureVectorPath(), options.getCentroidsPath(), options.getTrainDirPath(),
                options.getNumTestImagesPerClass(), options.getStart());
    }
}