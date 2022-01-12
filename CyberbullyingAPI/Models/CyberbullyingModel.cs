using NeuralNetwork.Core.MLP.Accuracies;
using NeuralNetwork.Core.MLP.Activations;
using NeuralNetwork.Core.MLP.Layers;
using NeuralNetwork.Core.MLP.Losses;
using NeuralNetwork.Core.Model;
using NeuralNetwork.Core.Text;

namespace CyberbullyingAPI.Models
{
    public class CyberbullyingModel
    {
        private static readonly string TextReaderPath = "D:/Datasets/glove/glove.twitter.27B.25d.txt";

        private static readonly int InputLength = 25;

        private static readonly TextReaderWordVector TextReader = new (TextReaderPath);

        private readonly Model Model;

        public CyberbullyingModel(string modelPath)
        {
            // Setup model
            var model = new Model();
            model.Set(loss: new LossCategoricalCrossentropy(), accuracy: new AccuracyClassification());
            model = InitModel(model);

            // Use saved params
            model.SetParametersFromFile(modelPath);

            Model = model;
        }

        private static Model InitModel(Model model)
        {
            // Layer 1
            model.Layers.Add(new LayerDense(InputLength, 256, weightsL2: 5e-4, biasesL2: 5e-4));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer 2
            model.Layers.Add(new LayerDense(256, 128));
            model.Layers.Add(new LayerDropout(0.2));
            model.Layers.Add(new ActivationReLU());

            // Layer Output
            model.Layers.Add(new LayerDense(128, 1));
            model.Layers.Add(new ActivationSigmoid());

            // Prepare model
            model.Prepare();

            return model;
        }

        public double Predict(string text)
        {
            var x = TextReader.GetWordVectors(text);

            var X = TextReaderWordVector.AverageWordVectors(x);

            if (X.Length == 0)
            {
                throw new Exception("Error: Could not read sentence");
            }

            var result = Model.Evaluate(X);

            if (result.Length == 0)
            {
                throw new Exception("Error: Could not make prediction");
            }

            return result[0];
        }
    }
}
