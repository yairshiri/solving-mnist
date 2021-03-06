using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = System.Random;

namespace ML
{
    public class Network
    {
        #region variables

        private static Random rand = new Random();
        private Layer[] _layers;
        private Loss _loss;
        private float learningRate;
        private float learningRate_decay = 0.999f;
        private Optimizer optimizer;
        private List<double> losses = new List<double>();
        private object gradientLock = new object();
        public Layer[] Layers
        {
            get => _layers;
            set => _layers = value;
        }
        
        public Loss Loss
        {
            get => _loss;
            set => _loss = value;
        }

        #endregion

        #region constructors

        public Network(Layer[] layers,float learningRate,int[] shape, Loss loss,Optimizer optimizer)
        {
            this.learningRate = learningRate; 
            Layers = layers;
            Layers[0].Init(shape);
            // initiating the layer sizes and weights
            for (int i = 1; i < layers.Length; i++)
            {
                shape = Layers[i - 1].outputShape;
                Layers[i].Init(shape);
            }
            Loss = loss;
            this.optimizer = optimizer;
            Optimizer.GetGrad  = backwards;
        }

        #endregion

        #region methods
        
        
        // a method that runs an input through the networks and gives an output
        public Tensor forwards(Tensor input)
        {
            Tensor ret= input;
            // looping through the layers
            for (int i = 0; i < Layers.Length; i++)
            {
                ret = Layers[i].Forwards(ret);
            }
            return ret;
        }
        
        // a method that computes the gradients for a single example 
        public (Tensor[],Tensor[]) backwards(Tensor features,Tensor labels)
        {
            // getting a prediction from the network
            Tensor pred = forwards(features);
            // finding the loss
            Tensor loss = Loss.Func((pred, labels));
            ((IList)losses).Add(loss.Value);
            int len = Math.Min(50, losses.Count);
            // getting the avarage of the last len losses
            double avgLoss = losses.Skip(losses.Count-len).Sum()/len;
            Debug.Log(pred.ToString()+labels.ToString()+loss.ToString()+"Avg loss in the last " +len + "runs: "+avgLoss);
            
            // a gradients array. the gradients will be applied after getting them. 
            // we put values in the array in the order we use and get them, meaning from the end to the start. 
            Tensor[] gradients = new Tensor[Layers.Length];
            // the gradients to be applied to the bias
            Tensor[] biasGradients = new Tensor[Layers.Length];
            // computing d_loss/d_y
            loss = Loss.FunctionDeriv((pred, labels));
            // running the backwards pass
            for (int i = Layers.Length-1; i >= 0; i--)
            {
                // getting the loss for the next pass and the gradients for the layer
                (loss, gradients[i],biasGradients[i]) = Layers[i].Backwards(loss);
                
            }
            // returning the gradiants
            return (gradients, biasGradients);
        }
        
        // a method that does a backwards pass though a network, with SGD
        public void backwards(Tensor[] data, Tensor[] labels)
        {
            (Tensor[] weightGrads, Tensor[] biasGrads) = optimizer.backwards(data, labels);
            ApplyGradients(weightGrads,biasGrads);
        }


        public void ApplyGradients(Tensor[] weightGrads, Tensor[] biasGrads)
        {
            //applying the gradients
            for (int i = Layers.Length-1; i >= 0; i--)
            {
                Layers[i].ApplyGradients(weightGrads[i],biasGrads[i]);
            }

        }


        public new string ToString()
        {
            string ret = "";
            for (int i = 0; i < Layers.Length; i++)
            {
                ret += Layers[i].ToString();
            }
            return ret;
        }
        #endregion
    }
}