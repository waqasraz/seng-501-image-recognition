package org.seng.image_classification.demos;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.IdentityFeatureExtractor;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.hadoop.sequencefile.TextBytesSequenceFileUtility;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.FImage2DoubleFV;
import org.openimaj.image.feature.ImageAnalyserFVFeatureExtractor;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.quantised.QuantisedKeypoint;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;

import org.openimaj.feature.SparseIntFV;
import java.io.ByteArrayInputStream;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import org.seng.image_classification.lookupClass;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import java.io.IOException;

/**
 * Created by waqas on 14/03/15.
 */
public class project {
    public static final String SEQUENCE_FILE_PATH = "/home/waqas/Data/part-r-00000";
    public static final String TEST_DATA_PATH = "/home/waqas/Data/temp";

    //Generate visual words
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift) {
        //List of sift features from training set
        List<LocalFeatureList<ByteDSIFTKeypoint>> siftFeatures = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        System.out.println("Sift train start");
        //For each image
        for (FImage img : sample) {
            System.out.println("image");
            //Get sift features
            pdsift.analyseImage(img);
            siftFeatures.add(pdsift.getByteKeypoints(0.005f));
        }

        System.out.println("Sift train done");

        //Reduce set of sift features for time
        if (siftFeatures.size() > 10000)
            siftFeatures = siftFeatures.subList(0, 10000);

        //Create a kmeans classifier with 600 categories (600 visual words)
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);

        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(siftFeatures);
        System.out.println("Start cluster");
        //Generate clusters (Visual words) from sift features.
        ByteCentroidsResult result = km.cluster(datasource);
        System.out.println("Cluster done");

        return result.defaultHardAssigner();
    }


    //Extract bag of visual words feature vector
    static class PHOWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public SparseIntFV extractFeature(FImage image) {

            //Get sift features of input image
            pdsift.analyseImage(image);

            //Gag of visual words histogram representation
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            //Bag of visual words for blocks and combine
            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            //Return normalised feature vector
            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds());
        }
    }

    public static void main( String[] args ) throws IOException
    {

        TextBytesSequenceFileUtility  reader = new TextBytesSequenceFileUtility(SEQUENCE_FILE_PATH, true);

        final GroupedDataset<String, ListDataset<SparseIntFV>, SparseIntFV> gds = new MapBackedDataset<String, ListDataset<SparseIntFV>, SparseIntFV>();
        lookupClass lookup = new lookupClass();

        for (final Map.Entry<Text, BytesWritable> kv : reader) {
            final MemoryLocalFeatureList<QuantisedKeypoint> features = MemoryLocalFeatureList.read(
                    new ByteArrayInputStream(kv.getValue().getBytes()), QuantisedKeypoint.class);

            final SparseIntFV vector = BagOfVisualWords.extractFeatureFromQuantised(features, 300);

            final String clz = lookup.get_Class(kv.getKey().toString());

            if (!gds.containsKey(clz))
                gds.put(clz, new ListBackedDataset<SparseIntFV>());
            gds.get(clz).add(vector);

        }


        LiblinearAnnotator<SparseIntFV, String> ann;
        HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        //IdentityFeatureExtractor<SparseIntFV> extractor = new IdentityFeatureExtractor<SparseIntFV>();
        //extractor.extractFeature(vector);
        ann = new LiblinearAnnotator<SparseIntFV, String>( hkm.createWrappedExtractor(new IdentityFeatureExtractor<SparseIntFV>()), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.out.println("Start training");
        ann.train(gds);
        System.out.println("Train done");

        //System.out.print(ann.classify(gds.getRandomInstance()).getPredictedClasses());
        VFSListDataset<FImage> testSet = new VFSListDataset<FImage>(TEST_DATA_PATH, ImageUtilities.FIMAGE_READER);

        //Dense sift pyramid
        DenseSIFT dsift = new DenseSIFT(5, 5);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(
                dsift, 6f, 8);

        //Assigner assigning sift features to visual word
        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(testSet, pdsift);

        //Feature extractor based on bag of visual words
        FeatureExtractor<SparseIntFV, FImage> extractor =new PHOWExtractor(pdsift,assigner);

        FImage image = ImageUtilities.readF(new File(TEST_DATA_PATH+"/66786387.jpg"));

        System.out.print(ann.classify(extractor.extractFeature(image)).getPredictedClasses());
    }

//    public SparseIntFV extractFeature(FImage image) {
//        DenseSIFT dsift = new DenseSIFT(5, 8);
//        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 8);
//
//        pdsift.analyseImage(image);
//
//        FeatureExtractor<SparseIntFV, FImage> extractor = new LocalFeatureExtractor<>()
//        extractor.extractFeature(image);
//
//        return null;
//    }
}
