namespace NeuralNetwork.Core.MLP
{
    // This is an abstract class, meaning that it is not meant to be instantiated on its own
    // It defines the properties and methods of its sub classes
    // It also inherits from the NetworkItem class with the generic set to NetworkMLP
    public abstract class NetworkMLP : NetworkItem<NetworkMLP>
    {
        // The 'override' keyword is used to override the base class implementation
        public override NetworkMLP Next { get; set; }
        public override NetworkMLP Prev { get; set; }

        public double[][] Inputs { get; set; }
        public double[][] Output { get; protected set; } // 'protected': It can only be set from this class, or its sub-classes
        public double[][] DInputs { get; set; }
    }
}
