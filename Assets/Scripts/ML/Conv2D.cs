using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace ML
{
    public class Conv2D: LearningLayer
    {
        #region variables

        private static double _learningRate = 1E-2;
        private int filterNum;
        private int strides;
        private int padding;
        private int[] kernelSize;
        private int skip;
        private int inputFilters=0;
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
                Assert.IsTrue(inputShape.SequenceEqual(value.Shape));
                _neuronActivations = value.Clone();
            }
        }
        #endregion

        #region constructors
        // all constructor need to get an activation function and a size
        
        // constructor with name
        public Conv2D(int[] shape,ActionLayer activation, int skip, int strides=0, int padding=0, string name="") : base(shape, activation,name)
        {
            this.skip = skip;
            this.strides = strides;
            this.padding = padding;
            outputShape = shape;
            Name = name;
            // we use outputshape[0] and inputshape[0] because the input is allways a vector with dense layers.
        }
        public Conv2D(int filterNum,int kernelHeight,int kernelWidth, int skip=1, int strides=1, int padding = 0, string activation="relu",string name="") : base(new int[]{}, activation,name)
        {
            kernelSize = new[] { kernelHeight, kernelWidth };
            this.filterNum = filterNum;
            this.skip = skip;
            this.strides = strides;
            this.padding = padding;
            Name = name;
        }
        #endregion

        #region methods
        // initiating the variables
        public override void Init(int[] inputShape)
        {
            base.Init(inputShape);
            inputFilters = inputShape[2];
            _neuronActivations = new Tensor(inputShape,name:Name + " neuron activations");
            _weights = new Tensor(new []{kernelSize[0],kernelSize[1],filterNum},name:Name+" Weights");
            outputShape = new[] { (inputShape[0] - kernelSize[0]+2*padding)/skip + 1, (inputShape[1] - kernelSize[1]+2*padding)/skip + 1, filterNum };
            // initiating the bias with a 0.001 value, will init with other values later (like we generate weights)
            Bias = new Tensor(filterNum,name:Name+" bias");
            //initiating the weight values with the xavier method
            double lim;
            switch (Activation)
            {
                case SigmoidLayer _:
                    //we use xavier init method for sigmoid
                    lim = Math.Sqrt(1.0 / (inputShape[0]*inputShape[1] + outputShape[0]*outputShape[1]));
                    break;
                case SeluLayer _:
                    // if we are using the selu activation layer we also want to use the lecun method
                    lim = Math.Sqrt(1.0 / inputShape[0]*inputShape[1]);
                    break;
                default:
                    // He method for all of the rest
                    lim = Math.Sqrt(2.0 / inputShape[0]*inputShape[1]);
                    break;
            }

            double WeightFunc() => Normal(0, lim);
            // generating the weights
            for (int i = 0; i < Weights.Height; i++)
            {
                for (int j = 0; j < Weights.Width; j++)
                {
                    for (int k = 0; k < filterNum; k++)
                    {
                        Weights[i][j][k].Value = WeightFunc();
                    }
                }
            }
        }
        
        // feedforward of a classical Dense layer
        protected override Tensor fPass(Tensor input)
        {
            // saving the input for the backwards pass
            _neuronActivations = new Tensor(input);
            Tensor ret = new Tensor(outputShape);
            // i loops over the output Height
            for (int i = 0; i < ret.Height; i++)
            {
                //j loops over the output Width
                for (int j = 0; j < ret.Width; j++)
                {
                    // l loops over the weight Height 
                    for (int l = 0; l < Weights.Height; l++)
                    {
                        // m loops over the weight width 
                        for (int m = 0; m < Weights.Width; m++)
                        {
                            // k loops over the output dimension
                            for (int k = 0; k < ret.Shape[2]; k++)
                            {
                                // n loops over the input dimension
                                for (int n = 0; n < inputFilters; n++)
                                {
                                    ret[i][j][k] += input[i*skip + l *  strides][j*skip + m * strides][n] *
                                                    Weights[l][m][k];
                                }
                            }
                        }
                    }
                    // adding the bias
                    ret[i][j] += Bias;
                }
            }
            return ret;
        }

        // backwards pass of a classical Dense layer
        protected override (Tensor, Tensor, Tensor) bPass(Tensor loss)
        {
            // verify that loss has the appropriate shape
            Assert.IsTrue(Enumerable.SequenceEqual(loss.Shape,outputShape));
            // creating the weight gradients vector
            Tensor wGrads = new Tensor(Weights.Shape);
            // creating the input gradients vector
            Tensor aGrads = new Tensor(inputShape);
            // the vector for the bias
            Tensor bGrads = new Tensor(Bias.Shape);
            // getting the gradients for the weights and the bias
            // l loops over the weight Height 
            for (int i = 0; i < wGrads.Height; i++)
            {
                for (int j = 0; j < wGrads.Width; j++)
                {
                    for (int l = 0; l < loss.Height; l++)
                    {
                        for (int m = 0; m < loss.Width; m++)
                        {
                            for (int k = 0; k < loss.Shape[2]; k++)
                            {
                                for (int n = 0; n < NeuronActivations.Shape[2]; n++)
                                {
                                    wGrads[i][j][k] += NeuronActivations[i*strides + l *  skip][j*strides + m * skip][n] * loss[l][m][k];
                                    aGrads[i * strides + l * skip][j * strides + m * skip][n] +=
                                        wGrads[i][j][k] * loss[l][m][k];
                                    bGrads[k] += loss[l][m][k];
                                }
                            }
                        }
                    }
                }
            }
            
            // we return AGrads because we use it as the loss for the prev layers.
            return (aGrads,wGrads,bGrads);
        }

        // the way we apply gradients in a fully connected layer, singular
        public override void ApplyGradients(Tensor wGrads,Tensor bGrads)
        {
            Assert.IsTrue(Enumerable.SequenceEqual(wGrads.Shape,Weights.Shape));
            Assert.IsTrue(Enumerable.SequenceEqual(bGrads.Shape,Bias.Shape));
            for (int k = 0; k < wGrads.Shape[2]; k++)
            {
                for (int j = 0; j < wGrads.Shape[1]; j++)
                {
                    for (int i = 0; i < wGrads.Shape[0]; i++)
                    {
                        Weights[i][j][k].Value -= wGrads[i][j][k].Value * _learningRate;
                    }
                }
                Bias[k].Value -= bGrads[k].Value * _learningRate;
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