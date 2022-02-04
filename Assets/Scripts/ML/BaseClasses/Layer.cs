using System;
using UnityEngine;
using UnityEngine.Assertions;
using Random = System.Random;

namespace ML
{
    public abstract class Layer
    {
        #region variables
        // input,output size is a vector because for example conv layers have 2d input/outputs.
        // output size is the shape of the output of this layer
        // input size is the shape of the output of the prev layer. or the input shape.
        public int[] outputShape;
        public int[] inputShape;
        public string Name { get; set; }
        
        // this remembers the last activations. Every layer needs to implemants this in a deferent way
        protected Tensor _neuronActivations;
        
        public Tensor NeuronActivations
        {
            get => _neuronActivations;
            set => _neuronActivations = value;
        }

        #endregion

        #region constructors
        // all constructor need to get an activation function and a size
        
        // constructor with name
        public Layer(int[] shape,string name = "")
        {
            Name = name;
            outputShape = shape;
        }
        #endregion
        
        #region methods

        public virtual void Init(int[] inputSize)
        {
            inputShape = inputSize;
        }


        public abstract Tensor Forwards(Tensor input);
        // return type is inputType,Matrix,vector because we return the gradients with respects to the inputs(inputtype) and the gradients to be applied
        // to the weights, which are metrices. Vector is for the bias
        public abstract (Tensor,Matrix,Vector) Backwards(Tensor loss);
        
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
        
        
        // wGrads is the gradients for the weights, and bGrads is the gradients for the bias 
        public abstract void ApplyGradients(Matrix wGrads,Vector bGrads);
        #endregion
    }
}