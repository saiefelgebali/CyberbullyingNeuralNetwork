using System;
using System.Collections.Generic;
using Accord.Math;
using NeuralNetwork.Core.Text;

namespace NeuralNetwork.WordEmbeddings
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Init text reader object
            Console.WriteLine("Reading vocabulary...");
            var textReader = new TextReaderWordVector("D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt");

            // User input loop
            string input = "";
            while (input != "QUIT")
            {
                Console.Write("Enter expression: ");

                input = Console.ReadLine();

                // Get and parse user input
                var expression = new PostfixExpression(input, (x) => {
                    var vector = textReader.GetWordVector(x);
                    if (vector != null) return vector;
                    return Vector.Zeros(25);
                });

                // Calculate closest word
                var result = ExpressionTree.EvaluateExpression(expression.PostfixTermsParsed);
                var word = GetClosestWord(result, textReader);

                Console.WriteLine($"=> {word}\n");
            }
        }

        static string GetClosestWord(double[] targetVector, TextReaderWordVector textReader)
        {
            // Closest match
            KeyValuePair<string, double[]>? closestWord = null;
            double closestDistance = double.PositiveInfinity;

            // Search for closest match
            foreach (var wordVec in textReader.Vocabulary)
            {
                var distance = Distance.Cosine(targetVector, wordVec.Value);
                if (distance < closestDistance)
                {
                    closestWord = wordVec;
                    closestDistance = distance;
                };
            };

            // Return closest match to vector
            return closestWord?.Key;
        }
    }
}
