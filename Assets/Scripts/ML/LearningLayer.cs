using UnityEngine.Assertions;

namespace ML
{
    public abstract class LearningLayer:Layer
    {
        #region Fields
        // every type of layer has weights
        private Matrix _weights = new Matrix();
        
        // this remembers the last activations. Every layer needs to implemants this in a deferent way
        protected Vector _neuronActivations;
        
        
        // the bias vector. it's length is the length of the output.
        protected Vector _bias;

        // using the copying constructor 
        public Matrix Weights
        {
            get => _weights;
            set => _weights = new Matrix(value);
        }
        
        public Activation Activation { get; set; }
        #endregion Fields
        
        #region Constructors
        public LearningLayer(int[] shape, Activation activationFunction, string name="") : base(shape,name)
        {
            Activation = activationFunction; 
        }
        #endregion

        #region Methods
        public abstract override Tensor Forwards(Tensor input);
        public abstract override (Tensor, Matrix, Vector) Backwards(Tensor loss);

        #endregion Methods

    }
}