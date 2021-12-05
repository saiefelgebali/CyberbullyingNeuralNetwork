using Accord.Math;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.Text
{
    /// <summary>
    /// Use a Word2Vec model to create a Vocabulary of words and their associated vectors.
    /// </summary>
    public class TextReaderWordVector
    {
        // 150 is the max absolute value in a word vector
        private readonly double normalizeMagnitude = 150;

        public Dictionary<string, double[]> Vocabulary;

        public TextReaderWordVector(string dictionary)
        {
            // Seperate lines
            string[] lines = File.ReadAllLines(dictionary);

            // Define word vectors from each line
            Vocabulary = new Dictionary<string, double[]>();

            for (int i = 0; i < lines.Length; i++)
            {
                var (word, vector) = Text.ParseWord(lines[i]);
                Vocabulary.Add(word, vector);
            }
        }

        public double[][] GetWordVectors(string text)
        {
            // Prepare text
            text = Text.Sanitize(text);
            string[] words = text.Split(' ');

            // Search word vectors
            double[][] vectors = new double[words.Length][];

            for (int i = 0; i < words.Length; i++)
            {
                double[] wordVector = GetWordVector(words[i]);
                if (wordVector != null) vectors[i] = wordVector;
            }

            // Remove null values
            vectors = vectors.Where(v => v != null).ToArray();

            return vectors;
        }

        public double[] GetWordVector(string word)
        {
            return Vocabulary.GetValueOrDefault(word);
        }

        public static double[] CombineWordVectors(double[][] vectors)
        {
            double[] result = new double[vectors.Columns()];

            // Combine words in a sentence by adding vectors together
            for (int i = 0; i < vectors.Rows(); i++)
            {
                for (int j = 0; j < vectors.Columns(); j++)
                {
                    result[j] += vectors[i][j];
                }
            }

            return result;
        }

        public double[] GetCombinedWordVectors(string text)
        {
            return CombineWordVectors(GetWordVectors(text));
        }

        public double[][] NormalizeWordVectors(double[][] wordVectors)
        {
            // 150 is the maximum absolute value in a vector
            return wordVectors.Divide(normalizeMagnitude);
        }        
        
        public double[] NormalizeWordVectors(double[] wordVectors)
        {
            // 150 is the maximum absolute value in a vector
            return wordVectors.Divide(normalizeMagnitude);
        }
    }
}
