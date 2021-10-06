using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeuralNetwork.Core.Text
{
    public class Text
    {
        public static (string, double[]) ParseWord(string wordDef)
        {
            // Split word definition
            string[] wordDefSplit = wordDef.Split(' ');

            // Word string is always first index
            string word = wordDefSplit[0];

            // Rest of definition is vector
            string[] stringVector = wordDefSplit.Skip(1).ToArray();

            // Convert vector to double vector
            double[] vector = new double[stringVector.Length];

            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] = Double.Parse(stringVector[i]);
            }

            // Return tuple
            return (word, vector);
        }

        public static string Sanitize(string text)
        {
            // Remove special characters
            // Allowed characters, a-z, A-Z, 0-9, <space>
            text = Regex.Replace(text, "[^a-zA-Z0-9 ]+", "", RegexOptions.Compiled);

            // Lowercase
            text = text.ToLower();

            return text;
        }
    }
}
