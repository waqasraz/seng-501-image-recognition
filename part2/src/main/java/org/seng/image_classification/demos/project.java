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
import org.seng.image_classification.lookupClass;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.io.IOException;

/**
 * Created by waqas on 14/03/15.
 */
public class project {
    public static final String SEQUENCE_FILE_PATH = "/home/waqas/Data/part-r-00000";

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
        //IdentityFeatureExtractor<SparseIntFV> extractor = new IdentityFeatureExtractor<SparseIntFV>();
        //extractor.extractFeature(vector);
        ann = new LiblinearAnnotator<SparseIntFV, String>(new IdentityFeatureExtractor<SparseIntFV>(), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.out.println("Start training");
        ann.train(gds);
        System.out.println("Train done");

    }
}
