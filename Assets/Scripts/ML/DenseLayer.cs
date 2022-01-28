using System;
using UnityEngine;
using UnityEngine.Assertions;
using Random = System.Random;

namespace ML
{
    public class DenseLayer: Layer<Vector,Vector>
    {
        #region variables
        private float _learningRate;
        
        

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

        public override Vector NeuronActivations
        {
            get => _neuronActivations;
            set
            {
                // need to check that the size of the neuron activations vector is good
                Assert.AreEqual(InputSize,value.Length);
                _neuronActivations = value;
            }
        }
        #endregion

        #region constructors
        // all constructor need to get an activation function and a size
        
        // constructor with name
        public DenseLayer(int size,Function<float,float> activationFunction,string name) : base(size, activationFunction, name)
        {
            OutputSize = size;
            Activation = activationFunction;
            Name = name;
        }
        // constructor without name
        public DenseLayer(int size,Function<float,float> activationFunction) : base(size, activationFunction)
        {
            OutputSize = size;
            Activation = activationFunction;
        }
        

        #endregion

        #region methods
        // initiating the variables
        public override void Init(int inputSize,float learningRate)
        {
            LearningRate = learningRate;
            //initiating the sizes and the vectors
            InputSize = inputSize;
            NeuronActivations = new Vector(inputSize);
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
        public override Vector Forwards(Vector input)
        {
            // saving the input for the backwards pass
            NeuronActivations = input;
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
                // passing through the activation function:
                ret[i] = Activation.Func(ret[i]);
            }

            return ret;
        }        
        // feedforward of a classical Dense layer, without the activation
        public Vector ForwardsRetNet(Vector input)
        {
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
        public override (Vector,Matrix,Vector) Backwards(Vector loss)
        {
            // verify that loss has the appropriate shape
            Assert.AreEqual(loss.Length,OutputSize);
            // creating the weight gradients vector
            Matrix wGrads = new Matrix(OutputSize,InputSize,0);
            // creating the input gradients vector
            Vector aGrads = new Vector(InputSize);
            // the vector for the bias
            Vector bGrads = new Vector(OutputSize);
            //updating the losses with dg/da (the gradient of the result of the activation with respect to the multiplicative sum)
            Vector forwardsResult = ForwardsRetNet(NeuronActivations);
            for (int i = 0; i < loss.Length; i++)
            {
                if (forwardsResult[i] == 0)
                    Debug.Log("moshe");
                loss[i] *= Activation.FunctionDeriv(forwardsResult[i]);
            }
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
            // we apply WGrads (weight += learning rate * grad)
            for (int i = 0; i < wGrads.Height; i++)
            {
                for (int j = 0; j < wGrads.Width; j++)
                {
                    // checking for wrong size gradients
                    if (i >= Weights.Height || j >= Weights.Width)
                    {
                        Debug.Log("oops");
                    }
                    Weights[i][j].Data -= wGrads[i][j] * LearningRate;
                }
            }
            // applying the bias gradients
            for (int i = 0; i < OutputSize; i++)
            {
                Bias[i] -= bGrads[i] * LearningRate;
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