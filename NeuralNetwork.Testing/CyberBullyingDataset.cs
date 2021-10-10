using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic.FileIO;
using NeuralNetwork.Core.Text;
using Accord.Math;
using NeuralNetwork.Core;

namespace NeuralNetwork.Testing
{
    public class CyberBullyingDataset
    {
        public static ((double[][], int[]), (double[][], int[])) PrepareCyberbullyingDataset(string path, TextReaderWordVector textReader)
        {
            var (X, y) = ParseCyberbullyingDataset(path, textReader);

            // Shuffle dataset
            int[] shuffledIndices = Utility.ShuffleIndices(X.Length);
            X = Utility.ShuffleArray(X, shuffledIndices);
            y = Utility.ShuffleArray(y, shuffledIndices);

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

        public static (double[][], int[]) ParseCyberbullyingDataset(string path, TextReaderWordVector textReader)
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
                if (fields.Length != 5)
                {
                    continue;
                }

                // Extract sample data
                string text = fields[0];
                double neutral = Double.Parse(fields[1]);
                double aggression = Double.Parse(fields[2]);
                double toxicity = Double.Parse(fields[3]);
                double racism = Double.Parse(fields[4]);
                double[] classList = new double[] { neutral, aggression, toxicity, racism };
                int targetClass = classList.IndexOf(classList.Max());

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
