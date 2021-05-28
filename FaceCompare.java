/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facecompare;

import cern.colt.Arrays;
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
import java.util.ArrayList;
import java.util.Collections;

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
        double[][] trainingData = new double[][]
        {
            {
                0.5, 2.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1
            },
            {
                0.7, 2.4, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9
            },
        };
        DoubleMatrix2D covarMatrix = makeCovarMatrix(trainingData);
        EigenvalueDecomposition eigens = new EigenvalueDecomposition(covarMatrix);

        DoubleMatrix2D eigenVectors = eigens.getV();
        DoubleMatrix1D eigenValues = eigens.getRealEigenvalues();
        DoubleMatrix2D eigenVectorsReduced = biggestEigenvalues(eigenValues, eigenVectors, 1);

        double[] means = calculateMeans(trainingData);
        double[] newData = new double[]
        {
            2.511, 2.411
        };

        DoubleMatrix2D transformedData = transform(newData, trainingData, means, eigenVectors);
        ArrayList<Double> closest = minEuclid(transformedData, trainingData);
        System.out.println(closest);;

    }

    public static ArrayList<Double> minEuclid(DoubleMatrix2D transformedData, double[][] trainingData)
    {
        DoubleMatrix2D allDistances = Statistic.distance(transformedData, EUCLID);
        ArrayList<Double> distances = new ArrayList<>();
        for (int i = 0; i < allDistances.rows(); i++)
        {
            distances.add(allDistances.getQuick(0, i));
        }
        distances.sort(null);
        double target = distances.get(1);

        boolean isFound = false;
        int index = 0;
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
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < trainingData.length; i++)
        {
            result.add(trainingData[i][index - 1]);
        }
        return result;
    }

    public static DoubleMatrix2D transform(double[] newData, double[][] trainingData, double[] means, DoubleMatrix2D eigenVectors)
    {
        int rows = trainingData.length;
        int columns = trainingData[0].length;

        DoubleMatrix2D result = new DenseDoubleMatrix2D(rows, columns + 1);
        for (int i = 0; i < newData.length; i++)
        {
            result.setQuick(i, 0, newData[i] - means[i]);
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 1; j <= columns; j++)
            {
                result.setQuick(i, j, trainingData[i][j - 1] - means[i]);
            }
        }

        DoubleMatrix2D transposedEigen = Algebra.DEFAULT.transpose(eigenVectors);

        return Algebra.DEFAULT.mult(transposedEigen, result);
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
}
