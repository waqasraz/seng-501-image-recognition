/**
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

public class GUI extends Frame {
    private Label lblCount;
    private TextField textField_OutPut;
    private JLabel imgLabel;
    private BufferedImage image;

    // Buttons
    private Button button_Start;
    private Button button_Quit;

    private int count = 0;     // Counter's value

    public GUI(){
        setLayout(new FlowLayout());

        image = null;
        try
        {
            image = ImageIO.read(new File("test.jpg"));
        }
        catch (Exception e)
        {
            e.printStackTrace();
            System.exit(1);
        }

        ImageIcon imageIcon = new ImageIcon(image);
        imgLabel = new JLabel();
        imgLabel.setIcon(imageIcon);
        add(imgLabel);

        lblCount = new Label("Output");
        add(lblCount);

        textField_OutPut = new TextField("", 20);
        textField_OutPut.setEditable(false);
        add(textField_OutPut);

        button_Start = new Button("Start");
        add(button_Start);
        button_Start.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Start the program
            }
        });

        button_Quit = new Button("Exit");
        add(button_Quit);
        button_Quit.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Quit the program
                System.exit(0);
            }
        });

        setTitle("Image Recognition");
        setSize(800, 600);

        setVisible(true);
    }

    public static void main(String[] args) {
        GUI app = new GUI();
    }
/*
    public void actionPerformed(ActionEvent evt) {
        ++count;
        textField_OutPut.setText(count + "");
    }
    */
}