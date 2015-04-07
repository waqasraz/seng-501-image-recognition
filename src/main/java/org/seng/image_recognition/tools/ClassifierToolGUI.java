package org.seng.image_recognition.tools; /**
 * Created by TConX on 22/03/2015.
 */

//package org.seng.image_classification;

import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import org.seng.image_recognition.core.data.CentroidsData;
import org.seng.image_recognition.core.data.FVData;
import org.seng.image_recognition.core.data.LocalFVData;
import org.seng.image_recognition.utils.ImageAnalysis;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.image.*;
import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.List;
import javax.imageio.*;
import javax.swing.*;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;

public class ClassifierToolGUI extends JFrame {
    private final LiblinearAnnotator<FVData, String> annotator;
    private final HardAssigner<byte[], float[], IntFloatPair> assigner;


    private JLabel label;
    private JTextField textField_OutPut;
    private JLabel imgLabel;
    private BufferedImage image;
    private FImage fImage;
    private JLabel statusbar;

    private JFrame GUI_Frame;
    private JPanel GUI_Panel;

    private JPanel image_Panel;
    private JPanel button_Panel;
    private JPanel status_Panel;

    private ImageIcon imageIcon;

    // Buttons
    private JButton button_Browse;
    private JButton button_Quit;

    private JFileChooser fc;
    private File file;

    private void switchImage(File imageFile){
        try
        {
            image = ImageIO.read(imageFile);

            //Classify image
            fImage = ImageUtilities.createFImage(image);
            DoubleFV fv = ImageAnalysis.extractFeatures(assigner, fImage);
            FVData fvdata = new LocalFVData(null, null, fv);
            ClassificationResult<String> classification = annotator.classify(fvdata);
            String result = new ArrayList<String>(classification.getPredictedClasses()).get(0);

            //Display result to user
            NumberFormat formatter = new DecimalFormat("#0.00");
            statusbar.setText("This image is a " + result + " (" + formatter.format(classification.getConfidence(result)) + "% sure)");
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.exit(1);
        }

        // TODO: Aspect Ratio PRESERVATION

        imageIcon = new ImageIcon(image);
        Image image = imageIcon.getImage();
        Image newimg = image.getScaledInstance(400, 400,  java.awt.Image.SCALE_SMOOTH);
        imageIcon = new ImageIcon(newimg);  // transform it back
        imgLabel.setIcon(imageIcon);
    }

    public ClassifierToolGUI(LiblinearAnnotator<FVData, String> ann, HardAssigner<byte[], float[], IntFloatPair> assigner){
        this.annotator = ann;
        this.assigner = assigner;

        fc = new JFileChooser();

        // Set up the main GUI
        GUI_Frame = new JFrame("Image Recognition");
        GUI_Panel = new JPanel();
        GUI_Panel.setLayout(new BorderLayout());
        GUI_Frame.getContentPane().add(GUI_Panel, "Center");

        button_Panel = new JPanel();
        button_Panel.setLayout(new FlowLayout());
        GUI_Panel.add(button_Panel, "Center");

        status_Panel = new JPanel();
        status_Panel.setLayout(new FlowLayout());
        GUI_Panel.add(status_Panel, "South");


        // Image Panel
        image_Panel = new JPanel();
        image = null;

        //imageIcon = new ImageIcon(image);
        imageIcon = new ImageIcon();
        imgLabel = new JLabel();
        imgLabel.setIcon(imageIcon);
        image_Panel.add(imgLabel);
        GUI_Frame.getContentPane().add(image_Panel, "North");

        statusbar = new JLabel();
        status_Panel.add(statusbar);

        button_Browse = new JButton("Select Image");
        button_Panel.add(button_Browse);
        button_Browse.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                int returnVal = fc.showOpenDialog(ClassifierToolGUI.this);

                if (returnVal == JFileChooser.APPROVE_OPTION) {
                    file = fc.getSelectedFile();
                    switchImage(file);
                } else {
                    // User canceled
                }
            }
        });

        button_Quit = new JButton("Exit");
        button_Panel.add(button_Quit);
        button_Quit.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Quit the program
                System.exit(0);
            }
        });

        GUI_Frame.setSize(600, 600);
        GUI_Frame.setVisible(true);
    }
}