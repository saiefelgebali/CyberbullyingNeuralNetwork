using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Core.Correlations;

namespace NeuralNetwork.Core.Layers
{
    public class LayerConvolution
    {
        // Shapes
        public int InputRows { get; private set; }
        public int InputColumns { get; private set; }
        public int KernelSize { get; private set; }
        public int KernelDepth { get; private set; }
        public int OutputRows { get; private set; }
        public int OutputColumns { get; private set; }

        // Layer variables
        public double[][][] Inputs { get; set; }
        public double[][] Bias { get; private set; }
        public double[][][] Kernels { get; private set; }
        public double[][][][] Output { get; set; }

        public LayerConvolution((int rows, int columns) inputSize, int kernelSize, int depth)
        {
            // Store layer shapes
            InputRows = inputSize.rows;
            InputColumns = inputSize.columns;
            KernelSize = kernelSize;
            KernelDepth = depth;
            OutputRows = InputRows - KernelSize + 1;
            OutputColumns = InputColumns - KernelSize + 1;

            if (KernelSize > InputColumns || KernelSize > InputRows)
            {
                throw new ArgumentException("Kernel must be smaller in size than input");
            }

            // Initialize random kernels
            Kernels = new double[depth][][];
            for (int i = 0; i < depth; i++)
            {
                Kernels[i] = Jagged.Random(kernelSize, kernelSize, -1.0, 1.0).Multiply(0.01);
            }

            // Initialize random bias
            Bias = Jagged.Zeros<double>(OutputRows, OutputColumns);
        }

        public void Forward(double[][][] input, bool training = false)
        {
            // Validate input shape
            if (input.Length < 1 || input[0].Rows() != InputRows || input[0].Columns() != InputColumns)
            {
                throw new ArgumentException("Input shape must match layer specifications");
            }

            Inputs = input;

            Output = new double[input.Length][][][];

            // Perform forward pass
            for (int sample = 0; sample < input.Length; sample++)
            {
                Output[sample] = new double[KernelDepth][][];
                for (int i = 0; i < KernelDepth; i++)
                {
                    Output[sample][i] = CrossCorrelation.ValidCrossCorrelation(input[sample], Kernels[i]);
                }
            }
        }

        public void Backward(double[][] dValues)
        {
            throw new NotImplementedException();
        }
    }
}
