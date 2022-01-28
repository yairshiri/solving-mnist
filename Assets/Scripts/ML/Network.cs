using System;
using System.Linq;
using UnityEngine;

namespace ML
{
    public class Network
    {
        #region variables

        private DenseLayer[] _layers;
        private Function<(Vector, Vector), Vector> _loss;
        public DenseLayer[] Layers
        {
            get => _layers;
            set => _layers = value;
        }
        
        public Function<(Vector, Vector), Vector> Loss
        {
            get => _loss;
            set => _loss = value;
        }

        #endregion

        #region constructors

        public Network(DenseLayer[] layers,float learningRate,int inputSize, Function<(Vector, Vector), Vector> loss)
        {
            
            Layers = layers;
            Layers[0].Init(inputSize,learningRate);
            // initiating the layer sizes and weights
            for (int i = 1; i < layers.Length; i++)
            {
                Layers[i].Init(Layers[i-1].OutputSize,learningRate);
            }
            Loss = loss;
        }

        #endregion

        #region methods
        
        
        // a method that runs an input through the networks and gives an output
        public Vector forwards(Vector input)
        {
            Vector ret= input;
            // looping through the layers
            for (int i = 0; i < Layers.Length; i++)
            {
                ret = Layers[i].Forwards(ret);
            }
            return ret;
        }
        
        // a method that does a backwards pass through the network
        public void backwards(Vector features,Vector labels)
        {
            // getting a prediction from the network
            Vector pred = forwards(features);
            // finding the loss
            Vector loss = Loss.Func((pred, labels));
            Debug.Log(pred.ToString()+labels.ToString()+loss.ToString());
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
                
            }
            
            //applying the gradients
            for (int i = Layers.Length-1; i >= 0; i--)
            {
             Layers[i].ApplyGradients(gradients[i],biasGradients[i]);
            }

            // printing the we have a new matrix, for seeing the gradients.
            //Debug.Log("Backwards pass ended!!!\n\n");
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