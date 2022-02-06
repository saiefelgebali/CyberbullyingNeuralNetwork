﻿using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Core.Optimizers;
using NeuralNetwork.Core.MLP;
using NeuralNetwork.Core.MLP.Losses;
using NeuralNetwork.Core.MLP.Layers;
using NeuralNetwork.Core.MLP.Activations;
using NeuralNetwork.Core.MLP.Accuracies;
using NeuralNetwork.Core.MLP.ActivationLoss;
using System.Text.Json;
using System.IO;

namespace NeuralNetwork.Core.Model
{
    public class Model
    {
        public LayerInput InputLayer { get; private set; }

        public List<LayerMLP> Layers { get; private set; }

        public List<LayerDense> TrainableLayers { get; private set; }

        public Activation OutputActivation { get; private set; }

        public Optimizer Optimizer { get; private set; }

        public Loss Loss { get; private set; }

        public Accuracy Accuracy { get; private set; }

        private ActivationSoftmaxLossCategoricalCrossentropy SoftmaxClassifierOutput;

        public Model(Loss loss, Optimizer optimizer, Accuracy accuracy)
        {
            // Init model components
            Layers = new List<LayerMLP>();
            Loss = loss;
            Optimizer = optimizer;
            Accuracy = accuracy;
        }

        public Model()
        {
            // Only create new list
            Layers = new List<LayerMLP>();
        }

        public void Set(Loss loss = null, Optimizer optimizer = null, Accuracy accuracy = null)
        {
            if (loss != null) Loss = loss;
            if (optimizer != null) Optimizer = optimizer;
            if (accuracy != null) Accuracy = accuracy;
        }

        public void Prepare()
        {
            // Create and set input layer
            InputLayer = new LayerInput();

            // Init new trainable layers list
            TrainableLayers = new List<LayerDense>();

            for (int i = 0; i < Layers.Count; i++)
            {
                LayerMLP layer = Layers[i];

                // If first layer
                if (i == 0)
                {
                    layer.Prev = InputLayer;
                    layer.Next = Layers[i + 1];
                }

                // Hidden layers
                else if (i < Layers.Count - 1)
                {
                    layer.Prev = Layers[i - 1];
                    layer.Next = Layers[i + 1];
                }

                // Output layer
                else if (i == Layers.Count - 1)
                {
                    layer.Prev = Layers[i - 1];
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

        public void Train((double[][], int[]) trainingData, (double[][], int[])? validationData = null,
            int epochs = 1, int batchSize = 0, int logFreq = 1)
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
                    double[][] batchX;
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
                    for (int i = 0; i < TrainableLayers.Count; i++)
                    {
                        LayerDense layer = TrainableLayers[i];
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
                    double[][] batchX;
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

        public double[][] Forward(double[][] X, bool training = false)
        {
            // Start network by applying input
            InputLayer.Forward(X, training);

            // Loop through all layers and perform a forward pass
            for (int i = 0; i < Layers.Count; i++)
            {
                LayerMLP layer = Layers[i];

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
                    LayerMLP layer = Layers[Layers.Count - 1 - i];

                    layer.Backward(layer.Next.DInputs);
                }
            }

            // Otherwise start backward pass in loss
            Loss.Backward(output, yTrue);

            // Call backward method through all layers in reverse
            for (int i = 0; i < Layers.Count; i++)
            {
                LayerMLP layer = Layers[Layers.Count - 1 - i];

                layer.Backward(layer.Next.DInputs);
            }
        }

        public double[][] Evaluate(double[][] input)
        {
            return Forward(input);
        }

        public double[] Evaluate(double[] input)
        {
            var paddedInput = new double[][] { input };
            return Forward(paddedInput)[0];
        }

        // Return array of (weights, biases),
        // for all trainable layers in model
        private LayerDenseParams[] GetParameters()
        {
            LayerDenseParams[] parameters = new LayerDenseParams[TrainableLayers.Count];

            for (int i = 0; i < TrainableLayers.Count; i++)
            {
                LayerDense layer = TrainableLayers[i];
                parameters[i] = layer.GetParameters();
            }

            return parameters;
        }

        // Set params for all trainable layers in model
        public void SetParameters(LayerDenseParams[] parameters)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                var layer = TrainableLayers[i];
                layer.SetParameters(parameters[i]);
            }
        }

        // Deserialize JSON object from file
        // Use saved params data
        public void SetParametersFromFile(string @path)
        {
            string modelParamsJson = File.ReadAllText(path);
            var parameters = JsonSerializer.Deserialize<LayerDenseParams[]>(modelParamsJson);

            SetParameters(parameters);
        }

        // Serialize params and save as JSON in file
        public void SaveParameters(string @path)
        {
            // Serialize object
            var parameters = GetParameters();
            var serializedModel = JsonSerializer.Serialize(parameters);

            // Save to file
            File.WriteAllText(@path, serializedModel);
        }        
    }
}
