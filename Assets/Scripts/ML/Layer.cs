using System;
using UnityEngine;
using UnityEngine.Assertions;
using Random = System.Random;

namespace ML
{
    public abstract class Layer<inputType,outputType>
    {
        #region variables
        // every type of layer has weights
        private Matrix _weights = new Matrix();

        // this remembers the last activations. Every layer needs to implemants this in a deferent way
        protected Vector _neuronActivations;
        
        // output size is the amount of neurons in this layer
        public int OutputSize { get; set; }
        // input size is the amount of neurons in the prev layer. or the input size.
        public int InputSize { get; set; }
        
        // the bias vector. it's length is the length of the output.
        protected Vector _bias;

        // using the copying constructor 
        public Matrix Weights
        {
            get => _weights;
            set => _weights = new Matrix(value);
        }

        public string Name { get; set; }

        public Function<float, float> Activation { get; set; }
        
        public virtual Vector NeuronActivations
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
        public Layer(int size,Function<float,float> activationFunction,string name)
        {
            OutputSize = size;
            Activation = activationFunction;
            Name = name;
        }
        // constructor without name
        public Layer(int size,Function<float,float> activationFunction)
        {
            OutputSize = size;
            Activation = activationFunction;
        }


        #endregion
        
        #region methods

        public abstract void Init(int inputSize, float learningRate);


        public abstract outputType Forwards(inputType input);
        // return type is inputType,Matrix,vector because we return the gradients with respects to the inputs(inputtype) and the gradients to be applied
        // to the weights, which are metrices. Vector is for the bias
        public abstract (inputType,Matrix,Vector) Backwards(outputType input);
        // wGrads is the gradients for the weights, and bGrads is the gradients for the bias 
        public abstract void ApplyGradients(Matrix wGrads,Vector bGrads);
        
        // same deal as with the tensors:
        // all layers may have names, so instead of implementing this (the name adding) for every claas we implement here
        // and use in other classes (with base.ToString())
        public new virtual string ToString()
        {
            
            string ret = "";
            if (Name != "")
                ret += Name+":\n";
            return ret;
        }

        #endregion
    }
}