package org.seng.image_classification.demos;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.IdentityFeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.hadoop.sequencefile.TextBytesSequenceFileUtility;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.quantised.QuantisedKeypoint;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;

import org.openimaj.feature.SparseIntFV;
import java.io.ByteArrayInputStream;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import org.openimaj.ml.annotation.Annotated;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.io.IOException;

/**
 * Created by waqas on 14/03/15.
 */
public class project {
    public static final String TRAINING_PATH = "/home/waqas/Documents/Project 501/imagedata/train";
    public static final String TEST_PATH = "/home/waqas/Documents/Project 501/imagedata/test";

    public static final String SEQUENCE_FILE_PATH = "/home/waqas/Data/part-r-00000";

    //Extract bag of visual words feature vector
    class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage image) {

            //Get sift features of input image
            pdsift.analyseImage(image);

            //Gag of visual words histogram representation
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            //Bag of visual words for blocks and combine
            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            //Return normalised feature vector
            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }


    public static void main( String[] args ) throws IOException
    {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> data_set = new VFSGroupDataset<FImage>(TRAINING_PATH, ImageUtilities.FIMAGE_READER);

        GroupedDataset<String, ListDataset<FImage>, FImage> data = GroupSampler.sample(data_set, data_set.size(), false);

        // Loading test dataset
        VFSListDataset<FImage> test_Set = new VFSListDataset<FImage>(TEST_PATH, ImageUtilities.FIMAGE_READER);


        TextBytesSequenceFileUtility  reader = new TextBytesSequenceFileUtility(SEQUENCE_FILE_PATH, true);

        GroupedDataset<Text, ListDataset<SparseIntFV>, SparseIntFV> gds = new MapBackedDataset<Text, ListDataset<SparseIntFV>, SparseIntFV>();

        //Map<Text, SparseIntFV> data_new = new HashMap<Text, SparseIntFV>();

        for (Map.Entry<Text, BytesWritable> kv : reader) {
            MemoryLocalFeatureList<QuantisedKeypoint> features = MemoryLocalFeatureList.read(new ByteArrayInputStream(kv.getValue().getBytes()), QuantisedKeypoint.class);
            SparseIntFV vector = BagOfVisualWords.extractFeatureFromQuantised(features, 200);
           // data_new.put(kv.getKey(), vector);
            //gds.put(kv.getKey(), vector.getVector());
            System.out.println(kv.getKey().toString());
        }

        //System.out.println(data_new.entrySet());



        LiblinearAnnotator<SparseIntFV, String> ann;
        //IdentityFeatureExtractor<SparseIntFV> extractor = new IdentityFeatureExtractor<SparseIntFV>();
        //extractor.extractFeature(vector);
        //ann = new LiblinearAnnotator<SparseIntFV, String>(new IdentityFeatureExtractor<SparseIntFV>(), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);


        //HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Uniform);
        //Gist geature extractor

        //FeatureExtractor<SparseIntFV, FImage> extractor = hkm.createWrappedExtractor()


        //Classifier for gist features
        //ann = new LiblinearAnnotator<SparseIntFV, String>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.out.println("Start training");
        //ann.train(data);
        System.out.println("Train done");

    }
}
