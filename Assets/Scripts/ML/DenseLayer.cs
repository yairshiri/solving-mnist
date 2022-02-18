using System;
using System.Linq;
using JetBrains.Annotations;
using UnityEngine;
using UnityEngine.Assertions;
using Random = System.Random;

namespace ML
{
    public class DenseLayer: LearningLayer
    {
        #region variables
        private static float _learningRate = 0.00001f;

        public int InputSize;
        public int OutputSize;
        // if this is true, we will use the Lecun weight init method instead of the xavier one.
        private bool useLeCun;
        

        public Tensor Bias
        {
            get => _bias;
            set => _bias = value;
        }

        public new Tensor NeuronActivations
        {
            get => _neuronActivations;
            set
            {
                // checking that value is a vector and has the same shape as its supposed to be
                Assert.IsTrue(Enumerable.SequenceEqual(inputShape,value.Shape));
                _neuronActivations = value.Clone();
            }
        }
        #endregion

        #region constructors
        // all constructor need to get an activation function and a size
        
        // constructor with name
        public DenseLayer(int shape,ActionLayer activation,string name="",bool useLeCun = false) : base(new []{shape}, activation,name)
        {
            outputShape = new []{shape};
            Name = name;
            // we use outputshape[0] and inputshape[0] because the input is allways a vector with dense layers.
            OutputSize = shape;
        }
        public DenseLayer(int shape,string activation,string name="",bool useLeCun = false) : base(new []{shape}, activation,name)
        {
            outputShape = new []{shape};
            Name = name;
            // we use outputshape[0] and inputshape[0] because the input is allways a vector with dense layers.
            OutputSize = shape;
        }
        #endregion

        #region methods
        // initiating the variables
        public override void Init(int[] inputShape)
        {
            //initiating the sizes and the vectors
            InputSize = inputShape[0];
            base.Init(inputShape);
            _neuronActivations = new Tensor(InputSize,Name + " neuron activations");
            _weights = new Matrix(OutputSize, InputSize,name:Name+" weights");
            // initiating the bias with a 0.001 value, will init with other values later (like we generate weights)
            Bias = new Tensor(size:OutputSize,name:Name+" bias");
            //initiating the weight values with the xavier method
            var rand = new Random();
            double lim =Math.Sqrt(6.0/ InputSize + OutputSize);
            // if we are using the selu activation layer we also want to use the lecun method
            if (useLeCun || Activation.GetType() == typeof(SeluLayer))
            {
                //we use the lecun init method instead of the xavier one:
                lim = Math.Sqrt(3.0/ InputSize);
            }
            // generating the weights
            for (int i = 0; i < Weights.Height; i++)
            {
                for (int j = 0; j < Weights.Width; j++)
                {
                    // generating a new value (float) between -lim and lim
                    Weights[i][j].Value = rand.NextDouble()*2*lim-lim;
                }
            }
            // generating the gradients
            for (int i = 0; i < Bias.Length; i++)
            {
                // we generate the bias like we generate weights
                Bias[i].Value = rand.NextDouble()*2*lim-lim;
            }
        }
        
        // feedforward of a classical Dense layer
        protected override Tensor fPass(Tensor input)
        {
            // saving the input for the backwards pass
            _neuronActivations = new Tensor(input);
            Tensor ret = new Tensor(OutputSize);
            for (int i = 0; i < OutputSize; i++)
            {
                // summing the products:
                for (int j = 0; j < InputSize; j++)
                {
                    ret[i] += Weights[i][j] * input[j];
                }
                // adding the bias
                ret[i] += Bias[i];
            }
            return ret;
        }

        // backwards pass of a classical Dense layer
        protected override (Tensor, Tensor, Tensor) bPass(Tensor loss)
        {
            // verify that loss has the appropriate shape
            Assert.AreEqual(loss.NumOfElements,OutputSize);
            // creating the weight gradients vector
            Matrix wGrads = new Matrix(OutputSize,InputSize,0);
            // creating the input gradients vector
            Tensor aGrads = new Tensor(InputSize);
            // the vector for the bias
            Tensor bGrads = new Tensor(OutputSize);
            // getting the gradients for the weights and the bias
            //looping through the output neurons
            for (int i = 0; i < wGrads.Height; i++)
            {   
                //looping through the input neurons
                for (int j = 0; j < wGrads.Width; j++)
                {
                    wGrads[i][j] += loss[i] * NeuronActivations[j];
                    aGrads[j] += loss[i] * Weights[i][j];// check multiplication works
                }
                //bgrads calculation
                bGrads[i] = loss[i];
            }
            //printing the weight gradients, I want to check for exploding gradients
            //Debug.Log(wGrads.ToString());
            
            // we return AGrads because we use it as the loss for the prev layers.
            return (aGrads,wGrads,bGrads);
        }

        // the way we apply gradients in a fully connected layer, singular
        public override void ApplyGradients(Tensor wGrads,Tensor bGrads)
        {
            // we apply WGrads (weight -= grad)
            for (int i = 0; i < wGrads.Shape[0]; i++)
            {
                for (int j = 0; j < wGrads.Shape[1]; j++)
                {
                    // checking for wrong size gradients
                    if (i >= Weights.Shape[0] || j >= Weights.Shape[1])
                    {
                        Debug.Log("oops");
                    }
                    Weights[i][j].Value -= wGrads[i][j].Value * _learningRate;
                }
            }
            // applying the bias gradients
            for (int i = 0; i < OutputSize; i++)
            {
                Bias[i].Value -= bGrads[i].Value * _learningRate ;
            }

        }
        
        
        public override string ToString()
        {
            // getting the name, if exists
            string ret = base.ToString();
            // adding the weights toString
            ret += Weights.ToString();
            ret+= Bias.ToString();
            return ret;
        }

        #endregion
    }
}