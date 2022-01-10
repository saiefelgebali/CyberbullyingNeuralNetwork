using NeuralNetwork.Core.MLP;
using NeuralNetwork.Core.MLP.Layers;
using NeuralNetwork.Core.CNN;
using NeuralNetwork.Core.CNN.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using NeuralNetwork.Core.MLP.Losses;
using NeuralNetwork.Core.MLP.Accuracies;
using NeuralNetwork.Core.MLP.ActivationLoss;
using NeuralNetwork.Core.MLP.Activations;
using NeuralNetwork.Core.Optimizers;

namespace NeuralNetwork.Core.Model
{
    public class ModelConvolutional
    {
        public List<LayerCNN> CNNLayers { get; private set; }
        public LayerFlatten FlattenLayer { get; private set; }
        public List<LayerMLP> MLPLayers { get; private set; }

        public List<LayerConvolution> CNNTrainableLayers { get; private set; }
        public List<LayerDense> MLPTrainableLayers { get; private set; }

        public Activation OutputActivation { get; private set; }
        public Loss Loss { get; private set; }
        public OptimizerSGD Optimizer { get; private set; }
        public Accuracy Accuracy { get; private set; }
        public ActivationSoftmaxLossCategoricalCrossentropy SoftmaxClassifierOutput { get; private set; }

        public ModelConvolutional(Loss loss, OptimizerSGD optimizer, Accuracy accuracy)
        {
            CNNLayers = new();
            MLPLayers = new();

            Loss = loss;
            Optimizer = optimizer;
            Accuracy = accuracy;
        }

        public void PrepareFlattenLayer()
        {
            // Setup flatten layer
            var cnnOutput = CNNLayers.Last();
            FlattenLayer = new LayerFlatten(cnnOutput.OutputDepth, cnnOutput.OutputRows, cnnOutput.OutputColumns);
        }

        public void PrepareLayers()
        {
            CNNTrainableLayers = new();
            MLPTrainableLayers = new();

            foreach (var layer in CNNLayers)
            {
                if (layer.GetType() == typeof(LayerConvolution)) CNNTrainableLayers.Add(layer as LayerConvolution);
            }
            foreach (var layer in MLPLayers)
            {
                if (layer.GetType() == typeof(LayerDense)) MLPTrainableLayers.Add(layer as LayerDense);
            }
        }

        public void PrepareOutput()
        {
            OutputActivation = MLPLayers.Last() as Activation;
            // Check for softmax and categorical crossentropy activation
            if (OutputActivation.GetType() == typeof(ActivationSoftmax) && Loss.GetType() == typeof(LossCategoricalCrossentropy))
            {
                SoftmaxClassifierOutput = new ActivationSoftmaxLossCategoricalCrossentropy();
            }

            Loss.TrainableLayers = MLPTrainableLayers;
        }

        public double[][] Forward(double[][][][] X, bool training = false)
        {
            // Forward pass CNN network
            CNNLayers.First().Forward(X);
            for (int i = 1; i < CNNLayers.Count; i++)
            {
                var layer = CNNLayers[i];
                var prevLayer = CNNLayers[i - 1];
                layer.Forward(prevLayer.Output);
            }

            // Flatten CNN output
            FlattenLayer.Forward(CNNLayers.Last().Output);

            // Forward pass MLP network
            MLPLayers.First().Forward(FlattenLayer.Output);
            for (int i = 1; i < MLPLayers.Count; i++)
            {
                var layer = MLPLayers[i];
                var prevLayer = MLPLayers[i - 1];
                layer.Forward(prevLayer.Output, training);
            }

            // Return outputp of final MLP layer
            return MLPLayers.Last().Output;
        }

        public void Backward(double[][] output, int[] yTrue)
        {
            // Use softmax classifier output if exists
            if (SoftmaxClassifierOutput != null)
            {
                SoftmaxClassifierOutput.Backward(output, yTrue);

                // Set DInputs of softmax output
                OutputActivation.DInputs = SoftmaxClassifierOutput.DInputs;

                // Call backward method through MLP Layers
                for (int i = 1; i < MLPLayers.Count; i++)
                {
                    LayerMLP layer = MLPLayers[MLPLayers.Count - 1 - i];

                    layer.Backward(MLPLayers[i + 1].DInputs);
                }

                // Backward on flatten layer
                FlattenLayer.Backward(MLPLayers.First().DInputs);

                // Call backward method through all CNN Layers
                CNNLayers.Last().Backward(FlattenLayer.DInputs);

                for (int i = 1; i < CNNLayers.Count; i++)
                {
                    LayerCNN layer = CNNLayers[CNNLayers.Count - 1 - i];

                    layer.Backward(CNNLayers[i + 1].DInputs);
                }
            }

            // Otherwise start backward pass in loss
            Loss.Backward(output, yTrue);

            // Call backward method through MLP Layers
            MLPLayers.Last().Backward(Loss.DInputs);
            for (int i = 1; i < MLPLayers.Count; i++)
            {
                LayerMLP layer = MLPLayers[MLPLayers.Count - 1 - i];

                layer.Backward(MLPLayers[MLPLayers.Count - i].DInputs);
            }

            // Backward on flatten layer
            FlattenLayer.Backward(MLPLayers.First().DInputs);

            // Call backward method through all CNN Layers
            CNNLayers.Last().Backward(FlattenLayer.DInputs);

            for (int i = 1; i < CNNLayers.Count; i++)
            {
                LayerCNN layer = CNNLayers[CNNLayers.Count - 1 - i];

                layer.Backward(CNNLayers[CNNLayers.Count - i].DInputs);
            }
        }

        public void Train((double[][][][] X, int[] y) trainingData, (double[][][][], int[])? validationData = null, int epochs = 1, int batchSize = 0, int logFreq = 1)
        {
            var (X, y) = trainingData;

            // Default value if batch size not set
            int trainSteps = 1;
            int validationSteps = 1;

            if (batchSize > 0)
            {
                // Calculate training steps
                trainSteps = Convert.ToInt32(X.Length / batchSize);

                if (trainSteps * batchSize < X.Length)
                {
                    trainSteps += 1;
                }

                if (validationData != null)
                {
                    var (XVal, _) = validationData.Value;

                    // Calculate validation steps
                    validationSteps = Convert.ToInt32(XVal.Length / batchSize);

                    if (validationSteps * batchSize < XVal.Length)
                    {
                        validationSteps += 1;
                    }
                }
            }

            // Main training loop
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Start epoch
                Console.WriteLine($"Epoch: {epoch}");

                // Reset accumulated values
                Loss.NewPass();
                Accuracy.NewPass();

                // Iterate over steps
                for (int step = 0; step < trainSteps; step++)
                {
                    double[][][][] batchX;
                    int[] batchY;

                    // If no batch size specified, use entire dataset
                    if (batchSize == 0)
                    {
                        batchX = X;
                        batchY = y;
                    }

                    // Otherwise slice a batch
                    else
                    {
                        batchX = X.Skip(step * batchSize).Take(batchSize).ToArray();
                        batchY = y.Skip(step * batchSize).Take(batchSize).ToArray();
                    }

                    // Perform a forward pass
                    double[][] output = Forward(batchX, training: true);

                    // Calculate loss
                    var (data_loss, reg_loss) = Loss.Calculate(output, batchY, regularization: true);
                    double loss = data_loss + reg_loss;

                    // Get predictions and calculate an accuracy
                    int[] predictions = OutputActivation.Predictions();
                    double accuracy = Accuracy.Calculate(predictions, batchY);

                    // Perform a backward pass
                    Backward(output, batchY);

                    // Update params using optimizer
                    Optimizer.PreUpdateParams();
                    foreach (var layer in CNNTrainableLayers)
                    {
                        Optimizer.UpdateParams(layer);
                    }
                    foreach (var layer in MLPTrainableLayers)
                    {
                        Optimizer.UpdateParams(layer);
                    }
                    Optimizer.PostUpdateParams();

                    // Show step summary
                    if (logFreq > 0 && step % logFreq == 0 || step - 1 == trainSteps)
                    {
                        Console.WriteLine($"Step: {step}");
                        Console.WriteLine($"Acc: {accuracy}");
                        Console.WriteLine($"Loss: {loss}");
                        Console.WriteLine($"DataLoss: {data_loss}");
                        Console.WriteLine($"RegLoss: {reg_loss}");
                        Console.WriteLine($"LR: {Optimizer.CurrentLearningRate}");
                        Console.WriteLine();
                    }
                }
                // Get and print epoch summary
                var (epochDataLoss, epochRegLoss) = Loss.CalculateAccumulated(regularization: true);
                double epochLoss = epochDataLoss + epochRegLoss;
                double epochAccuracy = Accuracy.CalculateAccumulated();

                Console.WriteLine($"Training");
                Console.WriteLine($"Acc: {epochAccuracy}");
                Console.WriteLine($"Loss: {epochLoss}");
                Console.WriteLine($"DataLoss: {epochDataLoss}");
                Console.WriteLine($"RegLoss: {epochRegLoss}");
                Console.WriteLine();
            }

            // Validation
            if (validationData != null)
            {
                var (XVal, yVal) = validationData.Value;

                // Reset accumulated
                Loss.NewPass();
                Accuracy.NewPass();

                // Iterate over steps
                for (int step = 0; step < validationSteps; step++)
                {
                    double[][][][] batchX;
                    int[] batchY;

                    // Use all samples
                    if (batchSize == 0)
                    {
                        batchX = XVal;
                        batchY = yVal;
                    }

                    // Slice a batch
                    else
                    {
                        batchX = XVal.Skip(step * batchSize).Take(batchSize).ToArray();
                        batchY = yVal.Skip(step * batchSize).Take(batchSize).ToArray();
                    }

                    // Perform forward pass
                    var output = Forward(batchX);

                    // Calculate loss
                    double loss = Loss.Calculate(output, batchY);

                    // Get predictions and calculate accuracy
                    int[] predictions = OutputActivation.Predictions();
                    Accuracy.Calculate(predictions, batchY);
                }

                double validationLoss = Loss.CalculateAccumulated();
                double validationAccuracy = Accuracy.CalculateAccumulated();

                // Show validation summary
                Console.WriteLine("Validation");
                Console.WriteLine($"Acc: {validationAccuracy}");
                Console.WriteLine($"Loss: {validationLoss}");
                Console.WriteLine();
            }
        }
    }
}
