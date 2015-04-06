package org.seng.image_recognition.tools; /**
 * Created by TConX on 22/03/2015.
 */

//package org.seng.image_classification;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.image.*;
import java.io.*;
import javax.imageio.*;
import javax.swing.*;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;

public class ClassifierToolGUI extends JFrame {
    private JLabel label;
    private JTextField textField_OutPut;
    private JLabel imgLabel;
    private BufferedImage image;
    private JLabel statusbar;

    private JFrame GUI_Frame;
    private JPanel GUI_Panel;

    private JPanel image_Panel;
    private JPanel results_Panel;
    private JPanel button_Panel;
    private JPanel status_Panel;

    private ImageIcon imageIcon;

    // Buttons
    private JButton button_Browse;
    private JButton button_Start;
    private JButton button_Quit;

    private JFileChooser fc;
    private File file;

    private void switchImage(File imageFile){
        try
        {
            image = ImageIO.read(imageFile);
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

    public ClassifierToolGUI(){

        fc = new JFileChooser();

        // Set up the main GUI
        GUI_Frame = new JFrame("Image Recognition");
        GUI_Panel = new JPanel();
        GUI_Panel.setLayout(new BorderLayout());
        GUI_Frame.getContentPane().add(GUI_Panel, "Center");

        // Non-image elements
        results_Panel = new JPanel();
        results_Panel.setLayout(new FlowLayout());
        GUI_Panel.add(results_Panel, "North");

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

        label = new JLabel("Output");
        results_Panel.add(label);

        textField_OutPut = new JTextField("", 20);
        textField_OutPut.setEditable(false);
        results_Panel.add(textField_OutPut);

        button_Start = new JButton("Start");
        button_Panel.add(button_Start);
        button_Start.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Start the program
                statusbar.setText("Start!");
            }
        });

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