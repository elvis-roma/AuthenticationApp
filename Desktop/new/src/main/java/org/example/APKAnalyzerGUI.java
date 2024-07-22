package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.jar.*;

public class APKAnalyzerGUI extends JFrame {

    private JLabel filePathLabel;
    private JTextField filePathTextField;
    private JButton analyzeButton;
    private JTextArea resultTextArea;

    public APKAnalyzerGUI() {
        setTitle("APK Analyzer");
        setSize(600, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        filePathLabel = new JLabel("APK File Path:");
        filePathTextField = new JTextField(30);
        analyzeButton = new JButton("Analyze APK");
        resultTextArea = new JTextArea(20, 50);
        resultTextArea.setEditable(false);

        JPanel inputPanel = new JPanel();
        inputPanel.add(filePathLabel);
        inputPanel.add(filePathTextField);
        inputPanel.add(analyzeButton);

        JScrollPane scrollPane = new JScrollPane(resultTextArea);

        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout());
        contentPane.add(inputPanel, BorderLayout.NORTH);
        contentPane.add(scrollPane, BorderLayout.CENTER);

        analyzeButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                String apkFilePath = filePathTextField.getText().trim();
                analyzeAPK(apkFilePath);
            }
        });
    }

    private void analyzeAPK(String apkFilePath) {
        // Your existing analyzeAPK method logic goes here
        // I'm omitting the method for brevity, but you should integrate it similarly
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                APKAnalyzerGUI analyzer = new APKAnalyzerGUI();
                analyzer.setVisible(true);
            }
        });
    }
}
