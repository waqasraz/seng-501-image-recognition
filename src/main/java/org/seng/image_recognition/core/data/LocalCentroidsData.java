package org.seng.image_recognition.core.data;


import org.openimaj.ml.clustering.ByteCentroidsResult;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

public class LocalCentroidsData implements CentroidsData {
    protected ByteCentroidsResult centroids;

    public LocalCentroidsData(ByteCentroidsResult centroids) {
        this.centroids = centroids;
    }

    public void write(String path) throws IOException {
        File file = new File(path);
        file.getParentFile().mkdirs();

        PrintWriter printWriter = new PrintWriter(file);

        this.centroids.writeASCII(printWriter);
        printWriter.close();
    }

    public static CentroidsData read(String path) throws IOException {
        Scanner in = new Scanner(new File(path));

        ByteCentroidsResult centroids = new ByteCentroidsResult();
        centroids.readASCII(in);
        CentroidsData data = new LocalCentroidsData(centroids);

        in.close();

        return data;
    }

    @Override
    public ByteCentroidsResult getCentroids() {
        return this.centroids;
    }
}
