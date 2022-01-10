﻿using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Core.Correlations;

namespace NeuralNetwork.Core.CNN.Layers
{
    public class LayerConvolution : LayerCNN
    {
        // Shapes
        public int SampleSize { get; private set; }
        public int InputRows { get; private set; }
        public int InputColumns { get; private set; }
        public int InputDepth { get; private set; }
        public int KernelSize { get; private set; }
        public int KernelDepth { get; private set; }
        public override int OutputDepth { get; protected set; }
        public override int OutputRows { get; protected set; }
        public override int OutputColumns { get; protected set; }

        // Forward propogation properties
        public double[][][][] Kernels { get; private set; }
        public double[][][] Biases { get; private set; }

        // Backward propogation properties
        public double[][][][] DKernels { get; set; }
        public double[][][] DBiases { get; set; }

        public LayerConvolution((int rows, int columns, int depth) inputDimensions, int kernelSize, int depth)
        {
            // Store layer shapes
            InputRows = inputDimensions.rows;
            InputColumns = inputDimensions.columns;
            InputDepth = inputDimensions.depth;
            KernelSize = kernelSize;
            KernelDepth = depth;
            OutputRows = InputRows - KernelSize + 1;
            OutputColumns = InputColumns - KernelSize + 1;
            OutputDepth = KernelDepth;

            if (KernelSize > InputColumns || KernelSize > InputRows)
            {
                throw new ArgumentException("Kernel must be smaller in size than input");
            }

            // Initialize random kernels
            Kernels = new double[KernelDepth][][][];
            for (int i = 0; i < KernelDepth; i++)
            {
                Kernels[i] = new double[InputDepth][][];
                for (int j = 0; j < InputDepth; j++)
                {
                    Kernels[i][j] = Jagged.Random(kernelSize, kernelSize, -1.0, 1.0).Multiply(0.01);
                }
            }

            // Initialize random bias
            Biases = new double[depth][][];
            for (int i = 0; i < depth; i++)
            {
                Biases[i] = Jagged.Zeros<double>(OutputRows, OutputColumns);
            }
        }

        public LayerConvolution((int rows, int columns, int depth) inputDimensions, double[][][][] initialKernels, double[][][] initialBiases)
        {
            // Store layer shapes
            InputRows = inputDimensions.rows;
            InputColumns = inputDimensions.columns;
            InputDepth = inputDimensions.depth;
            KernelDepth = initialKernels.Length;
            KernelSize = initialKernels[0][0].Rows();
            OutputRows = InputRows - KernelSize + 1;
            OutputColumns = InputColumns - KernelSize + 1;

            // Validate inputs
            if (initialKernels.Length < 1 || initialKernels[0].Length < 1)
            {
                throw new ArgumentException(
                    "initialKernels must be a 4D matrix." +
                    "Dimension 1 mus be the depth, " +
                    "Dimenson 2 must be the inputDepth"
                    );
            }

            if (KernelSize > InputColumns || KernelSize > InputRows)
            {
                throw new ArgumentException("Kernel must be smaller in size than input");
            }

            if (initialBiases.Length != KernelDepth)
            {
                throw new ArgumentException("Biases must be a 3D array with length of kernel depth");
            }

            if (initialBiases[0].Rows() != OutputRows || initialBiases[0].Columns() != OutputColumns)
            {
                throw new ArgumentException("Biases must be of same shape as output matrix");
            }

            // Initialize kernels with input
            Kernels = initialKernels;

            // Initialize random bias
            Biases = initialBiases;
        }

        public override void Forward(double[][][][] input)
        {
            // Validate input shape
            if (input.Length < 1 || input[0][0].Rows() != InputRows || input[0][0].Columns() != InputColumns)
            {
                throw new ArgumentException("Input shape must match layer specifications");
            }

            SampleSize = input.Length;

            Inputs = input;

            Output = new double[SampleSize][][][];

            // Perform forward pass
            for (int sample = 0; sample < SampleSize; sample++)
            {
                Output[sample] = new double[KernelDepth][][];
                // For each kernel layer
                for (int i = 0; i < KernelDepth; i++)
                {
                    Output[sample][i] = Jagged.Create<double>(OutputRows, OutputColumns);
                    // For each input
                    for (int j = 0; j < InputDepth; j++)
                    {
                        // Cross Correlation
                        var crossCorrelation = CrossCorrelation.ValidCrossCorrelation(input[sample][j], Kernels[i][j]).Add(Biases[i]);
                        Output[sample][i] = Output[sample][i].Add(crossCorrelation);
                    }
                }
            }
        }

        public override void Backward(double[][][][] dValues)
        {
            // Find DKernels
            DKernels = new double[KernelDepth][][][];

            for (int sample = 0; sample < dValues.Length; sample++)
            {
                for (int i = 0; i < KernelDepth; i++)
                {
                    if (sample == 0) DKernels[i] = new double[InputDepth][][];
                    for (int j = 0; j < InputDepth; j++)
                    {
                        if (j == 0) DKernels[i][j] = Jagged.Create<double>(OutputRows, OutputColumns);
                        DKernels[i][j] = CrossCorrelation.ValidCrossCorrelation(Inputs[sample][j], dValues[sample][i]);
                    }
                }
            }

            // Find DBiases
            for (int i = 0; i < dValues.Length; i++)
            {
                if (i == 0) DBiases = dValues[i];
                else
                {
                    for (int j = 0; j < dValues[i].Length; j++)
                    {
                        DBiases[j] = DBiases[j].Add(dValues[i][j]);
                    }
                }
            }

            // Find DInputs
            DInputs = new double[dValues.Length][][][];

            for (int sample = 0; sample < dValues.Length; sample++)
            {
                for (int i = 0; i < KernelDepth; i++)
                {
                    if (i == 0) DInputs[sample] = new double[InputDepth][][];
                    for (int j = 0; j < InputDepth; j++)
                    {
                        if (j == 0) DInputs[sample][j] = Jagged.Create<double>(InputRows, InputColumns);
                        DInputs[sample][j] = DInputs[sample][j].Add(CrossCorrelation.FullConvolultion(dValues[sample][j], Kernels[i][j]));
                    }
                }
            }
        }
    }
}
