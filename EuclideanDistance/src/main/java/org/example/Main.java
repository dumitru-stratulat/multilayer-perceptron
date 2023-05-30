package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Main {
    static final int ROW_OF_TARGET_VALUE = 7;
    static final int COLUMN_OF_TARGET_VALUE = 8;
    static final int TOTAL_ROWS = 2810;
    static final int TOTAL_INPUTS = 64;

    public static void main(String[] args) {
        int[][][] array;
        int[][][] arrayTest;
        //array to save inputs
        array = readingFile("cw2DataSet1.csv");
        arrayTest = readingFile("cw2DataSet2.csv");

        distance(array, arrayTest);
        distance(arrayTest, array);
    }

    public static void distance(int[][][] array1, int[][][] array2) {
        //7 and 8 are the target index of the values
        //constants
        final int totalInputRow = 7;
        final int totalInputColumn = 7;
        double distance;
        double minDistance;
        double minDistanceArray[] = new double[TOTAL_ROWS];
        //iterating throw dataset1 and dataset2
        for (int rowOfSet1 = 0; rowOfSet1 < TOTAL_ROWS; rowOfSet1++) {
            minDistance = Double.MAX_VALUE;
            ;
            for (int rowOfSet2 = 0; rowOfSet2 < TOTAL_ROWS; rowOfSet2++) {
                distance = 0;
                //iterating and calculating euclidian distance
                for (int row = 0; row <= totalInputRow; row++) {
                    for (int column = 0; column <= totalInputColumn; column++) {
                        distance += Math.pow((array1[rowOfSet1][row][column] - array2[rowOfSet2][row][column]), 2);
                    }
                }
                //calculating minimum distance
                if (distance < minDistance) {
                    minDistance = distance;
                    minDistanceArray[rowOfSet1] = array2[rowOfSet2][ROW_OF_TARGET_VALUE][COLUMN_OF_TARGET_VALUE];
                }
            }
        }
        int counter = 0;
        for (int i = 0; i < TOTAL_ROWS; i++) {
            if (array1[i][ROW_OF_TARGET_VALUE][COLUMN_OF_TARGET_VALUE] != minDistanceArray[i])
                counter++;
        }
        //calculating percentage
        double percentage = (TOTAL_ROWS - (double) counter) / (double) 2810 * 100;
        System.out.println("percentage=" + percentage);
    }

    public static int[][][] readingFile(String filename) {
        String line = "";
        String splitBy = ",";

        int[][][] array;
        int k = 0;
        array = new int[2810][8][9];
        try {
            //parsing a CSV file into BufferedReader class constructor
            BufferedReader br = new BufferedReader(new FileReader(filename));

            int lineCounter = 0;
            int digitCounter = 0;

            while ((line = br.readLine()) != null) {
                if (digitCounter == TOTAL_INPUTS) digitCounter = 0;
                //spliting every line
                String[] data = line.split(splitBy);
                //iterating and saving data in array
                for (int row = 0; row <= ROW_OF_TARGET_VALUE; row++)
                    for (int column = 0; column <= COLUMN_OF_TARGET_VALUE; column++) {
                        if (column == COLUMN_OF_TARGET_VALUE) {
                            array[k][row][column] = Integer.parseInt(data[TOTAL_INPUTS]);
                        } else {
                            array[k][row][column] = Integer.parseInt(data[digitCounter]);
                            digitCounter++;
                        }
                    }
                k++;
                if (lineCounter == TOTAL_ROWS) break;
                lineCounter++;
            }

        } catch (IOException err) {
            System.out.println(err);
        }
        return array;
    }
}