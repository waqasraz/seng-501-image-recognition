package org.seng.image_classification.demos;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.IdentityFeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.hadoop.sequencefile.TextBytesSequenceFileUtility;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.quantised.QuantisedKeypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LinearSVMAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import org.seng.image_classification.lookupClass;
import org.seng.image_classification.lookupClass_Testdata;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by waqas on 14/03/15.
 */
public class project_2 {
    public static final String SEQUENCE_FILE_PATH = "/home/waqas/Data/part-r-00000";
    public static final String SEQUENCE_FILE_PATH_TEST_DATA = "/home/waqas/Data/quantised-sift-testdata";


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

        //Working with test data set
        TextBytesSequenceFileUtility  reader_test = new TextBytesSequenceFileUtility(SEQUENCE_FILE_PATH_TEST_DATA, true);

        final GroupedDataset<String, ListDataset<SparseIntFV>, SparseIntFV> gds_test = new MapBackedDataset<String, ListDataset<SparseIntFV>, SparseIntFV>();
        lookupClass_Testdata lookup_test = new lookupClass_Testdata();

        for (final Map.Entry<Text, BytesWritable> kv : reader_test) {
            final MemoryLocalFeatureList<QuantisedKeypoint> features = MemoryLocalFeatureList.read(
                    new ByteArrayInputStream(kv.getValue().getBytes()), QuantisedKeypoint.class);

            final SparseIntFV vector = BagOfVisualWords.extractFeatureFromQuantised(features, 300);

            final String image_name = lookup_test.get_Image_Name(kv.getKey().toString());

            System.out.print(image_name+" ");
            System.out.print(ann.classify(vector).getPredictedClasses());
            System.out.println();

            gds_test.put(image_name, new ListBackedDataset<SparseIntFV>());

        }
    }

}
