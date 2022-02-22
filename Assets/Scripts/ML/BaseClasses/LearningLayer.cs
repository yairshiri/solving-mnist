using System;
using UnityEngine.Assertions;

namespace ML
{
    public abstract class LearningLayer:Layer
    {
        #region Fields
        // every type of layer has weights
        protected Tensor _weights;
        
        // this remembers the last activations. Every layer needs to implemants this in a deferent way
        protected new Tensor _neuronActivations;

        private ActionLayer _activation;
        
        // the bias vector. it's length is the length of the output.
        protected Tensor _bias;
        
        private static Random rand = new Random();


        // using the copying constructor 
        public Tensor Weights
        {
            get => _weights;
            set => _weights = new Tensor(value);
        }

        protected ActionLayer Activation
        {
            get => _activation;
            set => _activation = value;
        }
        #endregion Fields
        
        #region Constructors
        public LearningLayer(int[] shape,ActionLayer activation ,string name="") : base(shape,name)
        {
            Activation = activation;
        }
        public LearningLayer(int[] shape,string activation ,string name="") : base(shape,name)
        {
            Activation = GetActivation(activation);
        }
        #endregion

        #region Methods

        public override Tensor Forwards(Tensor input)
        {
            return Activation.Forwards(fPass(input));
        }

        public override (Tensor, Tensor, Tensor) Backwards(Tensor loss)
        {   
            return bPass(Activation.bPass(loss));
        }
        // a method that gets a name of an activation function and returns the Actionlayer associated with it
        private ActionLayer GetActivation(string name)
        {
            // we ignore cases
            name = name.ToLower();
            ActionLayer ret;
            switch (name)
            {
                case "relu": ret = new ReLULayer(outputShape);
                    break;
                case "sigmoid": ret = new SigmoidLayer(outputShape);
                    break;
                case "linear": ret = new LinearLayer(outputShape);
                    break;
                case "softmax": ret = new SoftMaxLayer(outputShape);
                    break;
                case "softrelu":ret = new SoftReLULayer(outputShape);
                    break;
                case "selu": ret = new SeluLayer(outputShape);
                    break;
                default:
                    throw new Exception("Activation name is invalid!!!");
            }
            // setting the name of the activation to the name of the learning layer plus the type of activation and the word activation
            // example for a Dense layer called d1 with a relu activation:
            // ret.Name would be d1 Relu Activation
            ret.Name = Name + " "+name+" Activation";
            ret.inputShape = outputShape;
            return ret;
        }


        protected abstract Tensor fPass(Tensor input);
        protected abstract (Tensor, Tensor, Tensor)  bPass(Tensor input);
        
        protected static double Normal(double mean, double stdDev)
        {
            double u1 = 1.0-rand.NextDouble(); 
            double u2 = 1.0-rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2); 
            double randNormal =
                mean + stdDev * randStdNormal; 
            return randNormal;
        }

        
        #endregion Methods

    }
}