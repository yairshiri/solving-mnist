using System;
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

        public Network(Layer[] layers,float learningRate,int inputSize, Loss loss)
        {
            this.learningRate = learningRate; 
            Layers = layers;
            int[] shape = { inputSize };
            Layers[0].Init(shape);
            // initiating the layer sizes and weights
            for (int i = 1; i < layers.Length; i++)
            {
                shape = Layers[i - 1].outputShape;
                Layers[i].Init(shape);
            }
            Loss = loss;
        }

        #endregion

        #region methods
        
        
        // a method that runs an input through the networks and gives an output
        public Tensor forwards(Vector input)
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
        public (Matrix[],Vector[]) backwards(Vector features,Vector labels)
        {
            // getting a prediction from the network
            Vector pred = new Vector( forwards(features));
            // finding the loss
            Tensor loss = Loss.Func((pred, labels));
            //Debug.Log(pred.ToString()+labels.ToString()+loss.ToString());
            
            // a gradients array. the gradients will be applied after getting them. 
            // we put values in the array in the order we use and get them, meaning from the end to the start. 
            Matrix[] gradients = new Matrix[Layers.Length];
            // the gradients to be applied to the bias
            Vector[] biasGradients = new Vector[Layers.Length];
            // computing d_loss/d_y
            loss = Loss.FunctionDeriv((pred, labels));
            // running the backwards pass
            for (int i = Layers.Length-1; i >= 0; i--)
            {
                // getting the loss for the next pass and the gradients for the layer
                (loss, gradients[i],biasGradients[i]) = Layers[i].Backwards(loss);
                if (gradients[i] != null)
                {
                    gradients[i] *= learningRate;
                    biasGradients[i] *= learningRate;
                }
            }
            // returning the gradiants
            return (gradients, biasGradients);

        }
        
        // a method that does a backwards pass though a network, with SGD
        public void backwards(Tensor[] data, Tensor[] labels,int sampleSize)
        {
            // put sampleSize random elements from data and labels into mini batches.
            Tensor[] data_batch = new Tensor[sampleSize];
            Tensor[] labels_batch = new Tensor[sampleSize];
            //put the selected items in the batch arrays
            int index = (int)Math.Floor(rand.NextDouble() * labels.Length);
            for (int i = 0; i < sampleSize; i++)
            {
                data_batch[i] = data[index];
                labels_batch[i] = labels[index];
                // getting another random index
                index = (int)Math.Floor(rand.NextDouble() * labels.Length);
            }

            Tensor[][] weightGrads = new Tensor[sampleSize][];
            Tensor[][] biasGrads= new Tensor[sampleSize][];
            // getting the gradiants
            for (int i = 0; i < sampleSize; i++)
            {
                (weightGrads[i], biasGrads[i]) = backwards((Vector)data_batch[i],(Vector)labels_batch[i]);
            }
            // computing the final grad:
            Tensor[] finalWeightGrad = new  Matrix[weightGrads[0].Length];
            Tensor[] finalBiasGrad = new  Vector[biasGrads[0].Length];
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                // adding the weight and bias gradiants
                finalWeightGrad[i] = new Matrix(weightGrads[0][i],true);
                finalBiasGrad[i] = new Vector(biasGrads[0][i],true);
                for (int j = 0; j < sampleSize; j++)
                {
                    finalWeightGrad[i] += (Matrix)weightGrads[j][i];
                    finalBiasGrad[i] += (Vector)biasGrads[j][i];
                }
            }
            ApplyGradients(finalWeightGrad,finalBiasGrad);
        }


        public void ApplyGradients(Tensor[] weightGrads, Tensor[] biasGrads)
        {
            //applying the gradients
            for (int i = Layers.Length-1; i >= 0; i--)
            {
                Layers[i].ApplyGradients((Matrix)weightGrads[i],(Vector)biasGrads[i]);
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