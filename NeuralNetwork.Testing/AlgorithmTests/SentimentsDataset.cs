using Accord.Math;
using Microsoft.VisualBasic.FileIO;
using NeuralNetwork.Core;
using NeuralNetwork.Core.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Testing.AlgorithmTests
{
    internal class SentimentsDataset
    {
        public static ((double[][], int[]), (double[][], int[])) PrepareSentimentsDataset(string path, TextReaderWordVector textReader)
        {
            var (X, y) = ParseSentimentsDataset(path, textReader);

            // Split dataset to training and validation
            double validationRatio = 0.2;
            int validationCount = Convert.ToInt32(validationRatio * X.Length);

            // Training data
            double[][] XTrain = X.Skip(validationCount).ToArray();
            int[] yTrain = y.Skip(validationCount).ToArray();

            // Validation data
            double[][] XVal = X.Take(validationCount).ToArray();
            int[] yVal = y.Take(validationCount).ToArray();

            // Return as tuple of tuples
            return ((XTrain, yTrain), (XVal, yVal));
        }

        public static (double[][], int[]) ParseSentimentsDataset(string path, TextReaderWordVector textReader)
        {
            // Start reading from dataset csv
            using var csvParser = new TextFieldParser(@path);

            csvParser.CommentTokens = new string[] { "#" };
            csvParser.SetDelimiters(new string[] { "," });
            csvParser.HasFieldsEnclosedInQuotes = true;

            // Skip column names
            csvParser.ReadLine();

            // Samples
            var sampleTextList = new List<double[]>();
            var sampleTargetList = new List<int>();

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
                    // Skip unreadable lines
                    continue;
                }

                // Validate field length
                if (fields.Length != 2) continue;

                // Extract sample data
                string text = fields[1];
                var targetClass = int.Parse(fields[0]);

                // Get vector
                double[] textVector = TextReaderWordVector.CombineWordVectors(textReader.GetWordVectors(text));

                // Validate text
                if (textVector.Length == 0)
                {
                    continue;
                }

                // Add sample to list
                sampleTextList.Add(textVector);

                // Add sample target to list
                sampleTargetList.Add(targetClass);
            }

            return (sampleTextList.ToArray(), sampleTargetList.ToArray());
        }
    }
}
