/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facecompare;

import java.util.Arrays;
import cern.colt.list.DoubleArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.doublealgo.Statistic;
import static cern.colt.matrix.doublealgo.Statistic.EUCLID;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import cern.jet.stat.Descriptive;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author super
 */
public class FaceCompare
{

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        File f = new File("C:\\Users\\super\\OneDrive\\Documents\\NetBeansProjects\\FaceCompare\\Images\\tutorial");
        HashMap<double[], String> data = convertDirectoryToArray(f);
        double[][] trainingData = new double[200 * 100][data.keySet().size()];
        int index = 0;
        for (double[] d : data.keySet())
        {
            for (int i = 0; i < d.length; i++)
            {
                trainingData[i][index] = d[i];
            }
            index++;
        }
        double[] means = calculateMeans(trainingData);
        for (int i = 0; i < trainingData.length; i++)
        {
            for (int j = 0; j < trainingData[i].length; j++)
            {
                trainingData[i][j] = trainingData[i][j] - means[i];
            }
        }
        for (double[] d : data.keySet())
        {
            for (int i = 0; i < d.length; i++)
            {
                d[i] -= means[i];
            }
        }
        DoubleMatrix2D trainingMatrix = new DenseDoubleMatrix2D(trainingData);
        DoubleMatrix2D trainingMatrixTransposed = Algebra.DEFAULT.transpose(trainingMatrix);

        DoubleMatrix2D covarMatrix = Algebra.DEFAULT.mult(trainingMatrixTransposed, trainingMatrix);
        EigenvalueDecomposition eigens = new EigenvalueDecomposition(covarMatrix);

        DoubleMatrix2D eigenVectors = eigens.getV();
        DoubleMatrix1D eigenValues = eigens.getRealEigenvalues();

        DoubleMatrix2D eigenFaces = Algebra.DEFAULT.mult(eigenVectors, trainingMatrixTransposed);

        DoubleMatrix2D weights = Algebra.DEFAULT.mult(trainingMatrixTransposed, Algebra.DEFAULT.transpose(eigenFaces));

        File test = new File("Images\\andrew0.png");
        HashMap<double[], String> newData = convertToArray(test);
        for (double[] d : newData.keySet())
        {
            double[] newDataCompare = d;
            DoubleMatrix2D transformedData = transform(newDataCompare, weights, means, eigenFaces);
            System.out.println(minEuclid(transformedData, trainingData, data));
        }
    }

    public static String minEuclid(DoubleMatrix2D transformedData, double[][] trainingData, HashMap<double[], String> data)
    {
        //error in this
        DoubleMatrix2D allDistances = Statistic.distance(transformedData, EUCLID);
        ArrayList<Double> distances = new ArrayList<>();
        for (int i = 0; i < allDistances.columns(); i++)
        {
            distances.add(allDistances.getQuick(0, i));
        }
        distances.sort(null);
        double target = distances.get(1);
        boolean isFound = false;
        int index = 1;
        while (!isFound)
        {
            if (allDistances.getQuick(0, index) == target)
            {
                isFound = true;
            } else
            {
                index++;
            }
        }
        double[] result = new double[200 * 100];
        for (int i = 0; i < trainingData.length; i++)
        {
            result[i] = (trainingData[i][index - 1]);
        }
        for (double[] d : data.keySet())
        {
            if (Arrays.equals(result, d))
            {
                return data.get(d);
            }
        }
        return null;
    }

    public static DoubleMatrix2D transform(double[] newData, DoubleMatrix2D weights, double[] means, DoubleMatrix2D eigenFaces)
    {
        weights = Algebra.DEFAULT.transpose(weights);
        int rows = weights.rows();
        int columns = weights.columns();
        DoubleMatrix2D result = new DenseDoubleMatrix2D(rows, columns + 1);
        //apply transformation to newData
        for (int i = 0; i < newData.length; i++)
        {
            newData[i] -= means[i];
        }
        DoubleMatrix2D data = new DenseDoubleMatrix2D(1, newData.length);
        for (int i = 0; i < newData.length; i++)
        {
            data.setQuick(0, i, newData[i]);
        }
        data = Algebra.DEFAULT.mult(data, Algebra.DEFAULT.transpose(eigenFaces));
        for (int i = 0; i < data.columns(); i++)
        {
            result.setQuick(i, 0, data.getQuick(0, i));
        }
        for (int i = 0; i < rows; i++)
        {
            for (int j = 1; j <= columns; j++)
            {
                result.setQuick(i, j, weights.getQuick(i, j - 1));
            }
        }
        return result;
    }

    public static DoubleMatrix2D makeCovarMatrix(double[][] trainingData)
    {
        int dimension = trainingData.length;
        DoubleMatrix2D result = new DenseDoubleMatrix2D(dimension, dimension);

        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                result.setQuick(i, j, Descriptive.covariance(new DoubleArrayList(trainingData[i]), new DoubleArrayList(trainingData[j])));
            }
        }
        return result;
    }

    public static double[] calculateMeans(double[][] trainingData)
    {
        double[] means = new double[trainingData.length];
        for (int i = 0; i < trainingData.length; i++)
        {
            double mean = 0;
            for (int j = 0; j < trainingData[i].length; j++)
            {
                mean += trainingData[i][j];
            }
            means[i] = mean / (trainingData[i].length);
        }
        return means;
    }

    public static DoubleMatrix2D biggestEigenvalues(DoubleMatrix1D eigenValues, DoubleMatrix2D eigenVectors, int values)
    {
        DoubleMatrix2D result = new DenseDoubleMatrix2D(eigenVectors.rows(), values);
        ArrayList<Double> sorted = new ArrayList<>();
        for (int i = 0; i < eigenValues.size(); i++)
        {
            sorted.add(eigenValues.get(i));
        }
        sorted.sort(null);
        Collections.reverse(sorted);
        for (int i = 0; i < values; i++)
        {
            double toFind = sorted.get(i);
            boolean found = false;
            int index = 0;
            while (index < eigenValues.size() && found == false)
            {
                if (eigenValues.get(index) == toFind)
                {
                    found = true;
                } else
                {
                    index++;
                }
            }
            for (int j = 0; j < eigenVectors.rows(); j++)
            {
                result.setQuick(j, i, eigenVectors.get(j, index));
            }
        }
        return result;
    }

    public static HashMap<double[], String> convertToArray(File f)
    {
        try
        {
            BufferedImage image = resize(ImageIO.read(f), 100, 200);
            int height = image.getHeight();
            int width = image.getWidth();
            ArrayList<Double> data = new ArrayList<>();
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    Color c = new Color(image.getRGB(j, i));
                    data.add((double) (c.getRed() * 256 * 256 + c.getGreen() * 256 + c.getBlue()));
                }
            }
            double[] converted = new double[height * width];
            for (int i = 0; i < data.size(); i++)
            {
                converted[i] = data.get(i);
            }
            HashMap<double[], String> result = new HashMap<>();
            result.put(converted, f.getName());
            return result;
        } catch (IOException e)
        {
            System.out.println("An error occurred processing this file: " + f.getName());
        }
        return null;
    }

    private static BufferedImage resize(BufferedImage img, int height, int width)
    {
        //"Java Resize Image to Fixed Width and Height Example." Memory Not Found, 24 Oct. 
        //2017, memorynotfound.com/java-resize-image-fixed-width-height-example/. 
        //Accessed 1 June 2021. 
        Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }

    public static HashMap<double[], String> convertDirectoryToArray(File dataSet)
    {
        File[] trainingSet = dataSet.listFiles();
        HashMap<double[], String> result = new HashMap<>();
        if (trainingSet.length > 0)
        {
            for (int i = 0; i < trainingSet.length; i++)
            {
                File f = trainingSet[i];
                HashMap<double[], String> imageData = convertToArray(f);
                for (double[] d : imageData.keySet())
                {
                    result.put(d, imageData.get(d));
                }
            }
        }
        return result;
    }
}
