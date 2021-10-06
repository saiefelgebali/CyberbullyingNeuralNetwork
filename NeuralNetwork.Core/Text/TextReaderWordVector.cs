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

        public double[][] GetWordVectors(string text, int wordsLength = 0)
        {
            // Prepare text
            text = Text.Sanitize(text);
            string[] words = text.Split(' ');

            // Search word vectors
            double[][] vectors;

            // Return entire vector
            if (wordsLength == 0)
            {
                wordsLength = words.Length;
                vectors = new double[words.Length][];
            }
            // Use words length 
            else if (wordsLength > 0)
            {
                vectors = new double[wordsLength][];
            }
            else
            {
                throw new Exception("Could not instantiate a new vectors array.");
            }

            for (int i = 0; i < wordsLength; i++)
            {
                double[] wordVector = Vocabulary.GetValueOrDefault(words[i]);
                if (wordVector != null) vectors[i] = wordVector;
            }

            // Remove null values
            vectors = vectors.Where(v => v != null).ToArray();

            // Pad arrays

            return vectors;
        }
    }
}
