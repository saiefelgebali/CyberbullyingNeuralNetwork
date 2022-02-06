using NeuralNetwork.Core.Text;
using NeuralNetwork.Testing.AlgorithmTests;

namespace NeuralNetwork.Testing
{
    internal class Program
    {
        readonly static TextReaderWordVector TextReader = new ("D:/Datasets/glove/glove.twitter.27B.25d.txt");

        static void Main(string[] args)
        {
            TrainCBModel();
        }

        // Train and test cyberbullying model
        static void TrainCBModel()
        {
            var modelPath = "D:/Projects/ml_models/cyberbullying_model.json";

            CyberbullyingAlgorithm.TrainCyberbullyingModel(TextReader, modelPath);
            CyberbullyingAlgorithm.TestCyberbullyingModel(TextReader, modelPath);
        }

        // Test saved cyberbullying model
        static void TestCBModel()
        {
            var modelPath = "D:/Projects/ml_models/cyberbullying_model.json";

            CyberbullyingAlgorithm.TestCyberbullyingModel(TextReader, modelPath);
        }

        // Train and test twitter sentiments model
        static void TrainTSModel()
        {
            var modelPath = "D:/Projects/ml_models/sentiments_model_1.json";

            SentimentsAlgorithm.TrainSentimentsModel(TextReader,modelPath);
            SentimentsAlgorithm.TestSentimentsModel(TextReader, modelPath);
        }
        
        // Test saved twitter sentiments model
        static void TestTSModel()
        {
            var modelPath = "D:/Projects/ml_models/sentiments_model_1.json";

            SentimentsAlgorithm.TestSentimentsModel(TextReader, modelPath);
        }
    }
}
