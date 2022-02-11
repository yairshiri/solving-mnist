using System;
using UnityEngine.Assertions;
using Math = System.Math;

namespace ML
{
    public class ActionLayer : Layer
    {
        #region Fields

        private Function<Tensor, Tensor> action;
        private bool useElementWise;

        #endregion Fields

        #region Constructors

        // Action layers (such as flatten) have no activation function
        public ActionLayer(int[] shape, Function<Tensor, Tensor> action, string name = "") : base(shape, name)
        {
            this.action = action;
            Name = name;
        }

        public ActionLayer(int[] shape, string name = "", bool useElementWise = true) : base(shape, name)
        {
            Name = name;
            this.useElementWise = useElementWise;
        }

        #endregion

        protected void Init(Function<Tensor, Tensor> action)
        {
            this.action = action;
        }

        public override Tensor Forwards(Tensor input)
        {
            // saving the input to the layer. used in the backprop
            NeuronActivations = input.Clone();
            // copying the input Tensor and applying the function to each element
            Tensor ret;
            if (useElementWise)
                ret = input.ElementWiseFunction(action.Func);
            else
                ret = action.Func(NeuronActivations);
            return ret;
        }

        public override (Tensor, Tensor, Tensor) Backwards(Tensor loss)
        {

            Tensor result;
            if (useElementWise)
                result = NeuronActivations.ElementWiseFunction(action.FunctionDeriv);
            else
                result = action.FunctionDeriv(NeuronActivations);
            // need to multiply loss by result
            return (Tensor.MatrixMult(loss,result), null, null);
        }

        public override void ApplyGradients(Tensor wGrads, Tensor bGrads)
        {
            // no learnable parameters to learn so nothing to see here -_-
        }
    }




    public class SigmoidLayer : ActionLayer
    {
        #region Fields

        #endregion Fields

        #region Constructors

        public SigmoidLayer(int[] shape, string name = "") : base(shape, name)
        {
            var sigmoid = new Function<Tensor, Tensor>(SigmoidFunc, SigmoidDeriv, "Sigmoid");
            Init(sigmoid);
        }

        #endregion Constructors

        #region Methods

        private static Tensor SigmoidFunc(Tensor x)
        {
            //the sigmoid function
            return new Tensor(1 / (1 + (float)Math.Exp(-x.Value)));
        }

        private static Tensor SigmoidDeriv(Tensor x)
        {
            //the sigmoid derivative
            return new Tensor(SigmoidFunc(x).Value * (1 - SigmoidFunc(x).Value));
        }

        #endregion Methods

    }



    public class SoftMaxLayer : ActionLayer
    {
        #region Fields

        private static float NOISE = 0.00001f;

        #endregion

        public SoftMaxLayer(int[] shape, string name = "") : base(shape, name, false)
        {
            var softmax = new Function<Tensor, Tensor>(softmaxFunc, softmaxDeriv, "softmax");
            Init(softmax);
        }

        #region Methods

        private static Tensor softmaxFunc(Tensor x)
        {
            // initializing a new vector with the length of x.
            Tensor ret = new Tensor(x.Length);
            double sum = 0;
            // finding the max value in x
            double max = x[0].Value;
            for (int i = 1; i < ret.Length; i++)
                if (x[i].Value > max)
                    max = x[i].Value;
            // every element is e to the power of the elements devided by the sum of e to the power of all the elements.
            // implementation from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            for (int i = 0; i < ret.Length; i++)
            {
                // x = e^x
                ret[i] = new Tensor(Math.Exp(x[i].Value - max));
                // adding to the sum
                sum += ret[i].Value;
            }

            // deviding by the sum
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i].Value /= sum;
                //we do this because we dont want to have 0s (for backprop), so we set a lower bound (NOISE)
                ret[i].Value = Math.Max(ret[i].Value, NOISE);// check if the recursive call works!!
            }

            return ret;
        }

        // softmax deriv from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        private Tensor softmaxDeriv(Tensor x)
        {
            //copying x and getting the softmax values for the deriv
            Tensor softmaxes = softmaxFunc(_neuronActivations);
            Matrix ret = new Matrix(x.Length, x.Length);
            int delta;
            // i is the input index, j is the output index
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < x.Length; j++)
                {
                    // the kronecker delta
                    if (i == j)
                        delta = 1;
                    else
                        delta = 0;
                    ret[i][j].Value = softmaxes[i].Value * (delta - softmaxes[j].Value);
                }
            }

            return ret;// check for matrix multiplication working
        }



        #endregion

    }



    public class ReLULayer : ActionLayer
    {
        #region Fields

        #endregion Fields

        #region Constructors

        public ReLULayer(int[] shape, string name = "") : base(shape, name)
        {
            var reLu = new Function<Tensor, Tensor>(reluFunc, reluDeriv, "ReLU");
            Init(reLu);
        }

        #endregion Constructors

        #region Methods

        private static Tensor reluFunc(Tensor x)
        {
            return new Tensor(Math.Max(x.Value, 0));
        }

        private static Tensor reluDeriv(Tensor x)
        {
            float ret = 0;
            // the relu derivative: 
            //{ x > 0 : 1}
            //{ x <= 0: 0}
            if (x.Value > 0)
                ret = 1;
            return new Tensor(ret);
        }

        #endregion Methods

    }


    public class LinearLayer : ActionLayer
    {
        #region Fields

        #endregion Fields

        #region Constructors

        public LinearLayer(int[] shape, string name = "") : base(shape, name)
        {
            var linear = new Function<Tensor, Tensor>(linearFunc, linearDeriv, "Linear");
            Init(linear);
        }

        #endregion Constructors

        #region Methods

        private static Tensor linearFunc(Tensor x)
        {
            return new Tensor(x);
        }

        private static Tensor linearDeriv(Tensor x)
        {
            return new Tensor(1.0);
        }

        #endregion Methods
    }
    public class SoftReLULayer : ActionLayer
    {
        #region Fields

        #endregion Fields

        #region Constructors

        public SoftReLULayer(int[] shape, string name = "") : base(shape, name)
        {
            var linear = new Function<Tensor, Tensor>(SoftReLUFunc, SoftReLUDeriv, "Linear");
            Init(linear);
        }

        #endregion Constructors

        #region Methods

        private static Tensor SoftReLUFunc(Tensor x)
        {
            return new Tensor(Math.Log(1+Math.Exp(x.Value)));
        }

        private static Tensor SoftReLUDeriv(Tensor x)
        {
            return new Tensor(1 / (1 + Math.Exp(-x.Value)));
        }

        #endregion Methods
    }
}