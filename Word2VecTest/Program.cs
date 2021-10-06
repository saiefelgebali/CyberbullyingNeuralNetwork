using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using NeuralNetwork.Core.Text;

namespace Word2VecTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Reading Word2Vec dictionary...");

            var textReader = new TextReaderWordVector("D:/Datasets/glove.twitter.27B/glove.twitter.27B.25d.txt");

            var text = textReader.GetWordVectors("i value time over money!!!!!");

            Console.WriteLine();
        }
    }
}
