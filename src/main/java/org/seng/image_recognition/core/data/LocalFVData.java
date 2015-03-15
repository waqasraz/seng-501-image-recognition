package org.seng.image_recognition.core.data;

import org.openimaj.feature.DoubleFV;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class LocalFVData extends LocalImageData implements FVData {
    protected DoubleFV features;

    public LocalFVData() {
    }

    public LocalFVData(String path, String type, DoubleFV features) {
        super(path, type);
        this.features = features;
    }

    @Override
    public void readASCII(Scanner in) throws IOException {
        this.path = in.next();
        this.type = in.next();

        String[] lines = in.nextLine().trim().split(" ");
        double[] values = new double[lines.length];
        for (int i = 0; i < lines.length; i++) {
            values[i] = Double.parseDouble(lines[i]);
        }

        this.features = new DoubleFV(values);
    }

    @Override
    public void writeASCII(PrintWriter out) throws IOException {
        out.write(this.path + " ");
        out.write(this.type + " ");

        for (int i = 0; i < this.features.values.length; i++) {
            out.print(this.features.values[i] + " ");
        }

        out.println();
    }

    public static List<FVData> readList(String path) throws IOException {
        List<FVData> keypoints = new ArrayList<FVData>();

        Scanner in = new Scanner(new File(path));
        while (in.hasNext()) {
            FVData data = new LocalFVData();
            data.readASCII(in);
            keypoints.add(data);
        }
        in.close();

        return keypoints;
    }

    public DoubleFV getFeatureVector() {
        return this.features;
    }
}
