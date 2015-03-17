package org.seng.image_classification;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.feature.IdentityFeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.hadoop.sequencefile.TextBytesSequenceFileUtility;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.quantised.QuantisedKeypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

/**
 * Simple class to read the contents of a Hadoop sequence file
 */
public class SequenceFileReader {
	public static final String SEQUENCE_FILE_PATH = "/home/waqas/Data/part-r-00000";

    /**
	 * Quick demo on how to read the sequence file
	 */
    public static void main( String[] args ) throws IOException {


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


