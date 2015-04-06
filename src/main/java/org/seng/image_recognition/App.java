package org.seng.image_recognition;


import org.apache.commons.lang.ArrayUtils;
import org.seng.image_recognition.tools.FVExtractorTool;
import org.seng.image_recognition.tools.KPExtractorTool;
import org.seng.image_recognition.tools.ResultsAnalyzerTool;
import org.seng.image_recognition.tools.TrainerTool;

public class App {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Please specify which tool to use...");
            System.exit(1);
        }

        //Extract specified tool from args
        String tool = args[0];
        args = (String []) ArrayUtils.removeElement(args, tool);

        if (tool.toLowerCase().equals("kpextractor")) {
            System.out.println("Running kpextractor tool...");
            KPExtractorTool.main(args);
        }
        else if (tool.toLowerCase().equals("trainer")) {
            System.out.println("Running trainer tool...");
            TrainerTool.main(args);
        }
        else if (tool.toLowerCase().equals("fvextractor")) {
            System.out.println("Running fvextractor tool...");
            FVExtractorTool.main(args);
        }
        else if (tool.toLowerCase().equals("analyzer")) {
            System.out.println("Running analyzer tool...");
            ResultsAnalyzerTool.main(args);
        }
        else {
            System.err.println("Unrecognized tool specified...");
            System.exit(1);
        }
    }
}
