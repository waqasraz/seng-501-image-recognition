package org.seng.image_recognition.core.data;


import org.openimaj.feature.DoubleFV;


public interface FVData extends ImageData {
    public DoubleFV getFeatureVector();
}
