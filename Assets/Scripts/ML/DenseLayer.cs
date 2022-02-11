using System;
using UnityEngine;
using UnityEngine.Assertions;
using Random = System.Random;

namespace ML
{
    public class DenseLayer: LearningLayer
    {
        #region variables
        private float _learningRate = 0.00001f;

        public int InputSize;
        public int OutputSize;
        

        public float LearningRate
        {
            get => _learningRate;
            set => _learningRate = value;
        }


        public Vector Bias
        {
            get => _bias;
            set => _bias = value;
        }

        public new Vector NeuronActivations
        {
            get => _neuronActivations;
            set
            {
                // checking that value is a vector
                Assert.AreEqual(value.Dimension,1);
                // need to check that the size of the neuron activations vector is good
                Assert.AreEqual(InputSize,value.Length);
                _neuronActivations = new Vector(value);
            }
        }
        #endregion

        #region constructors
        // all constructor need to get an activation function and a size
        
        // constructor with name
        public DenseLayer(int shape,ActionLayer activation,string name="") : base(new []{shape}, activation,name)
        {
            outputShape = new []{shape};
            Name = name;
            // we use outputshape[0] and inputshape[0] because the input is allways a vector with dense layers.
            OutputSize = shape;
        }
        public DenseLayer(int shape,string activation,string name="") : base(new []{shape}, activation,name)
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
            NeuronActivations = new Vector(InputSize);
            Weights = new Matrix(OutputSize, InputSize, 0);
            // initiating the bias with a 0.001 value, will init with other values later (like we generate weights)
            Bias = new Vector(OutputSize,"bias");
            //initiating the weight values with the xavier method
            var rand = new Random();
            float lim =(float)Math.Sqrt(6.0/ InputSize + OutputSize);
            // generating the weights
            for (int i = 0; i < Weights.Height; i++)
            {
                for (int j = 0; j < Weights.Width; j++)
                {
                    // generating a new value (float) between -lim and lim
                    Weights[i][j].Data = ((float)rand.NextDouble()*2*lim)-lim;
                }
            }
            // generating the gradients
            for (int i = 0; i < Bias.Length; i++)
            {
                // we generate the bias like we generate weights
                Bias[i] = ((float)rand.NextDouble()*2*lim)-lim;
            }
        }
        
        // feedforward of a classical Dense layer
        protected override Tensor fPass(Tensor input)
        {
            // saving the input for the backwards pass
            NeuronActivations = new Vector(input);
            Vector ret = new Vector(OutputSize);
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
        protected override (Tensor, Matrix, Vector) bPass(Tensor loss)
        {
            // verify that loss has the appropriate shape
            Assert.AreEqual(loss.Length,OutputSize);
            // creating the weight gradients vector
            Matrix wGrads = new Matrix(OutputSize,InputSize,0);
            // creating the input gradients vector
            Vector aGrads = new Vector(InputSize);
            // the vector for the bias
            Vector bGrads = new Vector(OutputSize);
            // getting the gradients for the weights and the bias
            //looping through the output neurons
            for (int i = 0; i < wGrads.Height; i++)
            {   
                //looping through the input neurons
                for (int j = 0; j < wGrads.Width; j++)
                {
                    wGrads[i][j].Data += loss[i] * NeuronActivations[j];
                    aGrads[j] += loss[i] * Weights[i][j].Data;
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
        public override void ApplyGradients(Matrix wGrads,Vector bGrads)
        {
            // we apply WGrads (weight -= grad)
            for (int i = 0; i < wGrads.Height; i++)
            {
                for (int j = 0; j < wGrads.Width; j++)
                {
                    // checking for wrong size gradients
                    if (i >= Weights.Height || j >= Weights.Width)
                    {
                        Debug.Log("oops");
                    }
                    Weights[i][j].Data -= wGrads[i][j] ;
                }
            }
            // applying the bias gradients
            for (int i = 0; i < OutputSize; i++)
            {
                Bias[i] -= bGrads[i] ;
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