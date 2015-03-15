package org.seng.image_recognition.demos;

import java.io.File;
import java.io.IOException;
import java.util.*;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;

import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import org.seng.image_recognition.core.data.*;
import org.seng.image_recognition.core.extractors.MockFVDataExtractor;
import org.seng.image_recognition.utils.BasicGroupedDataset;
import org.seng.image_recognition.utils.ImageAnalysis;


public class FeatureExtractionDemo {
    public static int NUM_TRAINING_IMAGES_PER_CLASS = 5;
    public static int NUM_TEST_IMAGES_PER_CLASS = 5;

    /**
     * First stage of the image recognition process. Writes each image path and its
     * corresponding classification type to a file.
     *
     * Input:
     *      - The root directory of the training data
     * Ouput:
     *      - file containing path/type pairs for each image:
     *          img1Path car
     *          img2Path car
     *          img3Path bicycle
     *          img4Path bicycle
     *          img5Path motorbike
     *          img6Path motorbike
     *          ....
     */
    public static void classifier() throws IOException {
        File imageRoot = new File("imagecl/train/");

        //Prepare training data
        List<TypeData> trainingData = new ArrayList<TypeData>();
        for (File folder : Arrays.asList(imageRoot.listFiles())) {
            if (folder.isDirectory()) {
                Iterable<File> files = Arrays.asList(folder.listFiles()).subList(0, NUM_TRAINING_IMAGES_PER_CLASS);
                for (File file : files) {
                    trainingData.add(new LocalTypeData(file.getAbsolutePath(), folder.getName()));
                }
            }
        }

        //Write classification data to file
        LocalImageData.writeList("tmp/typedata.txt", trainingData);
    }

    /**
     * Second stage of the image recognition process. Generates keypoints for each image.
     *
     * CAN BE PERFORMED WITH A MAPREDUCE TASK!
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
    public static void kpExtractor() throws IOException {
        //Read training data
        List<TypeData> trainingData = LocalTypeData.readList("tmp/typedata.txt");

        //Generate keypoints
        List<KPData> keypoints = new ArrayList<KPData>();
        for (TypeData type : trainingData) {
            System.out.println("Generating training image keypoints");

            AbstractDenseSIFT<FImage> sift = ImageAnalysis.getSift();
            sift.analyseImage(type.getImage());
            keypoints.add(new LocalKPData(type.getPath(), type.getType(), sift.getByteKeypoints(0.005f)));
        }

        //Write keypoints to file
        LocalImageData.writeList("tmp/keypoints.txt", keypoints);
    }

    /**
     * Third stage of the image recognition process. Trains the quantizer based on the given keypoints.
     *
     * CANNOT BE DONE AS A MAPREDUCE TASK! Must be done on a single machine
     *
     * Input:
     *      - file containing path/type/keypoint pairs for each image
     * Ouput:
     *      - file containing the training data centroids.
     */
    public static void trainer() throws IOException {
        //Read keypoints
        List<KPData> keypoints = LocalKPData.readList("tmp/keypoints.txt");

        //Train quantizer
        System.out.println("Training quantizer (this may take several minutes...)");
        ByteCentroidsResult centroids = ImageAnalysis.trainQuantiser(keypoints);

        LocalCentroidsData data = new LocalCentroidsData(centroids);
        data.write("tmp/centroids-100.ser");
    }

    /**
     * Fourth stage of the image recognition process. Trains the quantizer based on the given keypoints.
     *
     * CAN BE PERFORMED WITH A MAPREDUCE TASK!
     *
     * Input:
     *      - file containing path/type/keypoint pairs for each image
     *      - file containing the training data centroids.
     * Ouput:
     *      - file containing path/type/featurepoint pairs for each image
     *          img1Path car fp1 fp1 ...
     *          img2Path car fp1 fp2 ...
     *          img3Path bicycle fp1 fp2 ...
     *          img4Path bicycle fp1 fp2 ...
     *          img5Path motorbike fp1 fp2 ...
     *          img6Path motorbike fp1 fp2 ...
     *          ....
     */
    public static void fvExtractor() throws IOException {
        //Read keypoints
        List<KPData> keypoints = LocalKPData.readList("tmp/keypoints.txt");

        //Generate feature vectors
        List<FVData> features = new ArrayList<FVData>();
        for (KPData kpdata : keypoints) {
            System.out.println("Generating training image features");

            //Read assigner from file
            CentroidsData centroidsData = LocalCentroidsData.read("tmp/centroids.ser");
            HardAssigner<byte[], float[], IntFloatPair> assigner = centroidsData.getCentroids().defaultHardAssigner();

            DoubleFV fv = ImageAnalysis.extractFeatures(assigner, kpdata);
            features.add(new LocalFVData(kpdata.getPath(), kpdata.getType(), fv));
        }

        //Write features to file
        LocalImageData.writeList("tmp/features.txt", features);
    }

    /**
     * Fifth stage of the image recognition process. Reads the computed centroids and feature vectors and
     * uses them to classify images.
     *
     * In this demo, classified test images are used to determine the effectiveness of the image recognizer.
     *
     * Input:
     *      - file containing path/type/featurepoint pairs for each image
     *      - file containing the training data centroids.
     * Output:
     *      - The classification of the given test image
     */
    public static void recognizer() throws IOException {
        //Read features
        List<FVData> features = LocalFVData.readList("tmp/features.txt");

        CentroidsData centroidsData = LocalCentroidsData.read("tmp/centroids.ser");
        HardAssigner<byte[], float[], IntFloatPair> assigner = centroidsData.getCentroids().defaultHardAssigner();

        //Train annotator
        System.out.println("Training annotator");
        BasicGroupedDataset<FVData> trainingDataset = ImageAnalysis.groupFeatures(features);
        MockFVDataExtractor extractor = new MockFVDataExtractor();
    	LiblinearAnnotator<FVData, String> ann = new LiblinearAnnotator<FVData, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(trainingDataset);

        //Prepare test data
        File imageRoot = new File("imagecl/train/");
        List<FVData> testData = new ArrayList<FVData>();
        for (File folder : Arrays.asList(imageRoot.listFiles())) {
            if (folder.isDirectory()) {
                Iterable<File> files = Arrays.asList(folder.listFiles()).subList(
                        NUM_TRAINING_IMAGES_PER_CLASS, NUM_TRAINING_IMAGES_PER_CLASS + NUM_TEST_IMAGES_PER_CLASS);
                for (File file : files) {
                    System.out.println("Generating test image features");

                    DoubleFV fv = ImageAnalysis.extractFeatures(assigner, ImageUtilities.readF(file));

                    testData.add(new LocalFVData(file.getAbsolutePath(), folder.getName(), fv));
                }
            }
        }
        BasicGroupedDataset<FVData> testDataset = ImageAnalysis.groupFeatures(testData);

        //Test accuracy rate of annotator
        ClassificationEvaluator<CMResult<String>, String, FVData> eval =
                new ClassificationEvaluator<CMResult<String>, String, FVData>(
                        ann, testDataset, new CMAnalyser<FVData, String>(CMAnalyser.Strategy.SINGLE));
        Map<FVData, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);
    }

    /**
     * Demonstrates an outline of each step of the image recognition process using the local filesystem.
     *
     * The image recognition process has been broken up into 5 distinct stages. See the documentation of
     * each stage below for more info!
     */
    public static void main(String[] args) throws IOException {
        //Basic system verification
        if (System.getProperty("user.dir").contains(" ")) throw new RuntimeException("Your project folder path cannot contain spaces!");
        if (!new File("imagecl/").exists()) throw new RuntimeException("Please download and configure the demo images before proceeding");

        //Run all demo stages
        classifier();
        kpExtractor();
        trainer();
        fvExtractor();
        recognizer();
    }
}
