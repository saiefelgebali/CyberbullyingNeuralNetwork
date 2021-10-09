using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Core.Optimizers;
using NeuralNetwork.Core.Losses;
using NeuralNetwork.Core.Layers;
using NeuralNetwork.Core.Activations;
using NeuralNetwork.Core.Accuracies;
using NeuralNetwork.Core.ActivationLoss;

namespace NeuralNetwork.Core.Models
{
    public class Model
    {
        public LayerInput InputLayer { get; private set; }

        public List<NetworkLayer> Layers { get; private set; }

        public List<LayerDense> TrainableLayers { get; private set; }

        public Activation OutputActivation { get; private set; }

        public Optimizer Optimizer { get; private set; }

        public Loss Loss { get; private set; }

        public Accuracy Accuracy { get; private set; }

        private ActivationSoftmaxLossCategoricalCrossentropy SoftmaxClassifierOutput;

        public Model(Loss loss, Optimizer optimizer, Accuracy accuracy)
        {
            // Init model components
            Layers = new List<NetworkLayer>();
            Loss = loss;
            Optimizer = optimizer;
            Accuracy = accuracy;
        }

        public void Prepare()
        {
            // Create and set input layer
            InputLayer = new LayerInput();

            // Init new trainable layers list
            TrainableLayers = new List<LayerDense>();

            for (int i = 0; i < Layers.Count; i++)
            {
                NetworkItem layer = Layers[i];

                // If first layer
                if (i == 0)
                {
                    layer.Prev = InputLayer;
                    layer.Next = Layers[i + 1];
                }

                // Hidden layers
                else if (i < Layers.Count - 1)
                {
                    layer.Prev = Layers[i-1];
                    layer.Next = Layers[i+1];
                }

                // Output layer
                else if (i == Layers.Count - 1)
                {
                    layer.Prev = Layers[i-1];
                    layer.Next = Loss;
                    OutputActivation = layer as Activation;

                }

                // Check if layer is trainable
                if (layer.GetType() == typeof(LayerDense))
                {
                    TrainableLayers.Add(layer as LayerDense);
                }
            }
            // Check for softmax and categorical crossentropy activation
            if (OutputActivation.GetType() == typeof(ActivationSoftmax) && Loss.GetType() == typeof(LossCategoricalCrossentropy))
            {
                SoftmaxClassifierOutput = new ActivationSoftmaxLossCategoricalCrossentropy();
            }

            // Update loss object with trainable layers
            Loss.TrainableLayers = TrainableLayers;
        }

        public void Train(double[][] X, int[] y, (double[][], int[])? validationData = null, int epochs = 1, int logFreq = 1)
        {
            // Main training loop
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Perform a forward pass
                double[][] output = Forward(X, training: true);

                // Calculate loss
                var (data_loss, reg_loss) = Loss.Calculate(output, y, regularization: true);
                double loss = data_loss + reg_loss;

                // Get predictions and calculate an accuracy
                int[] predictions = OutputActivation.Predictions();
                double accuracy = Accuracy.Calculate(predictions, y);

                // Perform a backward pass
                Backward(output, y);

                // Update params using optimizer
                Optimizer.PreUpdateParams();
                for (int i = 0; i < TrainableLayers.Count; i++)
                {
                    LayerDense layer = TrainableLayers[i];
                    Optimizer.UpdateParams(layer);
                }
                Optimizer.PostUpdateParams();

                // Show summary
                if (epoch % logFreq == 0)
                {
                    Console.WriteLine($"Epoch: {epoch}");
                    Console.WriteLine($"Acc: {accuracy}");
                    Console.WriteLine($"Loss: {loss}");
                    Console.WriteLine($"data_loss: {data_loss}");
                    Console.WriteLine($"reg_loss: {reg_loss}");
                    Console.WriteLine($"LR: {Optimizer.CurrentLearningRate}");
                    Console.WriteLine();
                }
            }

            // Validation
            if (validationData != null)
            {
                var (XVal, yVal) = validationData.Value;

                // Perform forward pass
                var output = Forward(XVal);

                // Calculate loss
                double loss = Loss.Calculate(output, yVal);

                // Get predictions and calculate accuracy
                int[] predictions = OutputActivation.Predictions();
                double accuracy = Accuracy.Calculate(predictions, yVal);

                // Show summary
                Console.WriteLine("Validation");
                Console.WriteLine($"Acc: {accuracy}");
                Console.WriteLine($"Loss: {loss}");
            }
        }

        public double[][] Forward(double[][] X, bool training = false)
        {
            // Start network by applying input
            InputLayer.Forward(X, training);

            // Loop through all layers and perform a forward pass
            for (int i = 0; i < Layers.Count; i++)
            {
                NetworkLayer layer = Layers[i];

                layer.Forward(layer.Prev.Output, training);
            }

            // Return output of final layer
            return Layers.Last().Output;
        }

        public void Backward(double[][] output, int[] yTrue)
        {
            // Use softmax classifier output if exists
            if (SoftmaxClassifierOutput != null)
            {
                SoftmaxClassifierOutput.Backward(output, yTrue);

                // Set DInputs of softmax output
                OutputActivation.DInputs = SoftmaxClassifierOutput.DInputs;

                // Call backward method through all layers except last in reverse
                for (int i = 1; i < Layers.Count; i++)
                {
                    NetworkLayer layer = Layers[Layers.Count - 1 - i];

                    layer.Backward(layer.Next.DInputs);
                }
            }

            // Otherwise start backward pass in loss
            Loss.Backward(output, yTrue);

            // Call backward method through all layers in reverse
            for (int i = 0; i < Layers.Count; i++)
            {
                NetworkLayer layer = Layers[Layers.Count - 1 - i];

                layer.Backward(layer.Next.DInputs);
            }
        }
    }
}
