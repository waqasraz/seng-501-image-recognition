package org.seng.image_recognition.core.data;

import org.apache.commons.codec.DecoderException;
import org.apache.commons.codec.binary.Hex;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;


public class LocalKPData extends LocalImageData implements KPData {
    protected LocalFeatureList<ByteDSIFTKeypoint> keypoints;

    public LocalKPData() {
    }

    public LocalKPData(String path, String type, LocalFeatureList<ByteDSIFTKeypoint> keypoints) {
        super(path, type);
        this.keypoints = keypoints;
    }

    @Override
    public void readASCII(Scanner in) throws IOException {
        this.path = in.next();
        this.type = in.next();

        List<String> strings = Arrays.asList(in.nextLine().trim().split("\t"));

        List<ByteDSIFTKeypoint> points = new ArrayList<ByteDSIFTKeypoint>();
        for (String string : strings) {
            try {
                Scanner keypointScanner = new Scanner(string.trim());
                float x = Float.parseFloat(keypointScanner.next());
                float y = Float.parseFloat(keypointScanner.next());
                float energy = Float.parseFloat(keypointScanner.next());

                byte[] descriptor = Hex.decodeHex(keypointScanner.next().toCharArray());

                points.add(new ByteDSIFTKeypoint(x, y, descriptor, energy));
            } catch (DecoderException e) {
                throw new RuntimeException("Could not decode hex data " + string);
            }
        }
        this.keypoints = new MemoryLocalFeatureList<ByteDSIFTKeypoint>(points);
    }

    @Override
    public void writeASCII(PrintWriter out) throws IOException {
        out.write(this.path + " ");
        out.write(this.type + " ");

        for (ByteDSIFTKeypoint point : this.keypoints) {
            out.write(point.x + " " + point.y + " " + point.energy + " ");
            out.write(Hex.encodeHexString(point.descriptor));
            out.write("\t");

        }

        out.println();
    }

    public static List<KPData> readList(String path) throws IOException {
        List<KPData> keypoints = new ArrayList<KPData>();

        Scanner in = new Scanner(new File(path));
        while (in.hasNext()) {
            KPData data = new LocalKPData();
            data.readASCII(in);
            keypoints.add(data);
        }
        in.close();

        return keypoints;
    }

    @Override
    public LocalFeatureList<ByteDSIFTKeypoint> getKeypoints() {
        return this.keypoints;
    }
}
