//using Accord.Math;
//using MathNet.Numerics;
//using MathNet.Numerics.LinearAlgebra;
//using NeuralNetwork.Core.Model;
//using NeuralNetwork.Core.Accuracies;
//using NeuralNetwork.Core.Activations;
//using NeuralNetwork.Core.Layers;
//using NeuralNetwork.Core.Losses;
//using NeuralNetwork.Core.Optimizers;
//using System.Linq;

//namespace NeuralNetwork.Testing.AlgorithmTests
//{
//    public class SpiralDataset
//    {
//        public static void SpiralModelAlgorithm()
//        {
//            // Generate data
//            var (X, y) = GenerateSpiralData(1000, 3);
//            var validationData = GenerateSpiralData(10, 3);

//            // Create model
//            var loss = new LossCategoricalCrossentropy();
//            var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-5);
//            var accuracy = new AccuracyClassification();
//            var model = new Model(loss, optimizer, accuracy);

//            model.Layers.Add(new LayerDense(X.Columns(), 16, weightsL2: 5e-4, biasesL2: 5e-4));
//            model.Layers.Add(new LayerDropout(0.2));
//            model.Layers.Add(new ActivationReLU());

//            model.Layers.Add(new LayerDense(16, 3));
//            model.Layers.Add(new ActivationSoftmax());

//            // Prepare model
//            model.Prepare();

//            // Train model
//            model.Train((X, y), validationData, epochs: 1000, batchSize: 500, logFreq: 10000);
//        }

//        // Define dataset
//        public static (double[][] X, int[] y) GenerateSpiralData(int points, int classes)
//        {
//            var M = Matrix<double>.Build; //shortcut to Matrix builder
//            var V = Vector<double>.Build; //shortcut to Vector builder

//            //build vectors of size points*classesx1 for y, r and theta
//            var Y = V.Dense(points * classes); //at this point this is full of zeros
//            for (int j = 0; j < classes; j++)
//            {
//                var y_step = V.DenseOfArray(Generate.Step(points * classes, 1, (j + 1) * points));
//                Y = Y + y_step;
//            }
//            var r = V.DenseOfArray(Generate.Sawtooth(points * classes, points, 0, 1));
//            var theta = 4 * (r + Y) + (V.DenseOfArray(Generate.Standard(points * classes)) * 0.2);
//            var sin_theta = theta.PointwiseSin();
//            var cos_theta = theta.PointwiseCos();


//            double[][] X = M.DenseOfColumnVectors(r.PointwiseMultiply(sin_theta), r.PointwiseMultiply(cos_theta)).ToArray().ToJagged();

//            // convert y values to ints, and use one-hot vectors
//            int[] y = Y.Select((val) => (int)val).ToArray();

//            return (X, y);
//        }

//        //// Test code
//        //static void Algorithm1()
//        //{
//        //    int batchSize = 1000;

//        //    // Generate data
//        //    var (X, y) = SpiralDataset.GenerateSpiralData(batchSize, 3);

//        //    // Init neural network
//        //    var dense1 = new LayerDense(X.Columns(), 16, weightsL2: 5e-4, biasesL2: 5e-4);
//        //    var dropout1 = new LayerDropout(0.2);
//        //    var activation1 = new ActivationReLU();

//        //    var dense2 = new LayerDense(16, 3);
//        //    var lossActivation = new ActivationSoftmaxLossCategoricalCrossentropy();

//        //    lossActivation.Loss.TrainableLayers = new[] { dense1, dense2 };

//        //    //var optimizer = new OptimizerSGD(learningRate: 1, decay: 1e-3, momentum: 0.5);
//        //    //var optimizer = new OptimizerAdaGrad(decay: 1e-4);
//        //    //var optimizer = new OptimizerRMSProp(learningRate: 0.02, decay: 1e-5, rho: 0.999);
//        //    var optimizer = new OptimizerAdam(learningRate: 0.05, decay: 5e-5);

//        //    // Start Training
//        //    for (int epoch = 0; epoch < 10001; epoch++)
//        //    {
//        //        // Forward pass
//        //        dense1.Forward(X);
//        //        activation1.Forward(dense1.Output);
//        //        dropout1.Forward(activation1.Output);
//        //        dense2.Forward(dropout1.Output);
//        //        var (dataLoss, regLoss) = lossActivation.Forward(dense2.Output, y);
//        //        double loss = dataLoss + regLoss;

//        //        // Backward pass
//        //        lossActivation.Backward(lossActivation.Output, y);
//        //        dense2.Backward(lossActivation.DInputs);
//        //        dropout1.Backward(dense2.DInputs);
//        //        activation1.Backward(dropout1.DInputs);
//        //        dense1.Backward(activation1.DInputs);

//        //        // Update params
//        //        optimizer.PreUpdateParams();
//        //        optimizer.UpdateParams(dense1);
//        //        optimizer.UpdateParams(dense2);
//        //        optimizer.PostUpdateParams();

//        //        // Calculate accuracy
//        //        //double accuracy = Loss.Accuracy(lossActivation.Output, y);


//        //        if (epoch % 100 == 0)
//        //        {
//        //            //Console.Clear();
//        //            Console.WriteLine($"Epoch: { epoch }");
//        //            Console.WriteLine($"Data Loss: { dataLoss }");
//        //            Console.WriteLine($"Reg Loss: { regLoss }");
//        //            Console.WriteLine($"Loss: { loss }");
//        //            //Console.WriteLine($"Accuracy: { accuracy }");
//        //            Console.WriteLine($"LR: {optimizer.CurrentLearningRate}");
//        //            Console.WriteLine();
//        //            //Console.ReadKey();
//        //        }
//        //    }

//        //    // Validation
//        //    {
//        //        var (XTest, yTest) = SpiralDataset.GenerateSpiralData(100, 3);

//        //        dense1.Forward(XTest);
//        //        activation1.Forward(dense1.Output);
//        //        dense2.Forward(activation1.Output);
//        //        var (data_loss, reg_loss) = lossActivation.Forward(dense2.Output, yTest);
//        //        double testLoss = data_loss + reg_loss;
//        //        //double testAccuracy = Loss.Accuracy(lossActivation.Output, yTest);

//        //        Console.WriteLine($"Validation");
//        //        Console.WriteLine($"Loss: { testLoss }");
//        //        Console.WriteLine($"Accuracy: { testAccuracy }");
//        //        Console.WriteLine();
//        //    }
//        //}
//    }
//}
