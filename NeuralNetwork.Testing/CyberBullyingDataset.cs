using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic.FileIO;
using NeuralNetwork.Core.Text;
using Accord.Math;

namespace NeuralNetwork.Testing
{
    class Sample
    {
        public string Text { get; private set; }
        public double Aggression { get; private set; }
        public double Toxicity { get; private set; }
        public double Racism { get; private set; }
        public int[] GroundTruthVector { get; set; }

        // Init a new sample
        public Sample(string text, double aggression, double toxicity, double racism)
        {
            Text = text;
            Aggression = aggression;
            Toxicity = toxicity;
            Racism = racism;

            // Init ground truh vector with default 0
            GroundTruthVector = new int[4] { 0, 0, 0, 0 };
            double[] y = new double[4] { 1-Aggression-Toxicity-Racism, Aggression, Toxicity, Racism };
            double yMax= y.Max();

            // If yMax > 0.5, use as yTrue
            // Use one-hot encoding to save value
            if (yMax > 0.5)
            {
                int yTrue = y.IndexOf(yMax);
                GroundTruthVector[yTrue] = 1;
            }
        }
    }

    internal class CyberBullyingDataset
    {
        public Sample[] Samples { get; private set; }

        public CyberBullyingDataset(string path)
        {
            // Init samples list
            List<Sample> samples = new();

            // Start reading from dataset csv
            using var csvParser = new TextFieldParser(@path);

            csvParser.CommentTokens = new string[] { "#" };
            csvParser.SetDelimiters(new string[] { "," });
            csvParser.HasFieldsEnclosedInQuotes = true;

            // Skip column names
            csvParser.ReadLine();

            while (!csvParser.EndOfData)
            {
                // Read each row's fields
                string[] fields;
                try
                {
                     fields = csvParser.ReadFields();
                }
                catch
                {
                    continue;
                }

                // Validate field length
                if (fields.Length != 4)
                {
                    continue;
                }

                // Extract sample data
                string text = fields[0];
                double aggression = Double.Parse(fields[1]);
                double toxicity = Double.Parse(fields[2]);
                double racism = Double.Parse(fields[3]);

                // Add sample to list
                samples.Add(new Sample(text, aggression, toxicity, racism));
            }

            // Save samples as an array
            Samples = samples.ToArray();
        }

        /// <summary>
        /// X is the array of word vector samples.
        /// y is the ground truth index between 0-3.
        /// </summary>
        /// <example>
        /// For all y values,
        /// 0: Neutral
        /// 1: Aggression
        /// 2: Toxicity
        /// 3: Racism
        /// </example>
        public (double[][], int[]) PrepareDataset(TextReaderWordVector textReader, int wordVecDimensions)
        {
            // Prepare samples and ground truth
            List<double[]> x = new();
            List<int> y = new();

            for (int i = 0; i < Samples.Length; i++)
            {
                // Combine word vectors
                double[] vector = TextReaderWordVector.CombineWordVectors(textReader.GetWordVectors(Samples[i].Text));

                // Weed out unsuitable data
                if (vector.Length != wordVecDimensions)
                {
                    continue;
                }

                // Add to list
                x.Add(vector);
                // Sample y index as ground truth value
                int yMax = Samples[i].GroundTruthVector.Max();
                int yIndex = Samples[i].GroundTruthVector.IndexOf(yMax);
                y.Add(yIndex);
            }

            // Normalise Dataset
            double[][] X = x.ToArray();
            double xMax = X.Abs().Max();

            X = X.Divide(xMax);

            return (X, y.ToArray());
        }
    }
}
