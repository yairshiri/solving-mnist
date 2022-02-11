using System;
using UnityEngine.Assertions;
using Math = System.Math;

namespace ML
{
    public class ActionLayer:Layer
    {
        #region Fields
        private Function<Tensor,Tensor> action;
        private bool useElementWise;
        #endregion Fields
        #region Constructors
        // Action layers (such as flatten) have no activation function
        public ActionLayer(int[] shape,Function<Tensor,Tensor> action, string name="") : base(shape,name)
        {
            this.action = action;
            Name = name;
        }
        public ActionLayer(int[] shape ,string name="",bool useElementWise = true) : base(shape,name)
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

        public override (Tensor, Matrix, Vector) Backwards(Tensor loss)
        {

            Tensor result;
            if (useElementWise)
                result = NeuronActivations.ElementWiseFunction(action.FunctionDeriv);
            else
                result = action.FunctionDeriv(NeuronActivations);
                // need to multiply loss by result
            return (result.ElementWiseMultiply(loss),null,null);
        }
        public override void ApplyGradients(Matrix wGrads, Vector bGrads)
        {
            // no learnable parameters to learn so nothing to see here -_-
        }
    }
}



namespace ML
{
    public class SigmoidLayer:ActionLayer
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
            return new Scalar(1/(1+(float)Math.Exp(-x.Data)));
        }

        private static Tensor SigmoidDeriv(Tensor x)
        {
            //the sigmoid derivative
            return new Scalar(SigmoidFunc(x).Data*(1-SigmoidFunc(x).Data));
        }

        #endregion Methods

    }
}

namespace ML
{
    public class SoftMaxLayer:ActionLayer
    {
        #region Fields

        private static float NOISE = 0.00001f;

        #endregion
        
        public SoftMaxLayer(int[] shape, string name = "") : base(shape,name,false)
        {
            var softmax = new Function<Tensor, Tensor>(softmaxFunc, softmaxDeriv,"softmax");
            Init(softmax);
        }

        #region Methods

        private static Tensor softmaxFunc(Tensor x) 
        {
            // initializing a new vector with the length of x.
            Vector ret = new Vector(x.Length);
            float sum = 0;
            // finding the max value in x
            float max = x[0];
            for (int i = 1; i < ret.Length; i++)
                if (x[i] > max)
                    max = x[i];
            // every element is e to the power of the elements devided by the sum of e to the power of all the elements.
            // implementation from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            for (int i = 0; i < ret.Length; i++)
            {   
                // x = e^x
                ret[i] = (float)Math.Exp(x[i]-max);
                // adding to the sum
                sum += ret[i];
            }
            // deviding by the sum
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] /= sum;
                //we do this because we dont want to have 0s (for backprop), so we set a lower bound (NOISE)
                ret[i] = Math.Max(ret[i], NOISE);
            }
            return ret;
        }
        // softmax deriv from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        private Tensor softmaxDeriv(Tensor x)
        {
            //copying x and getting the softmax values for the deriv
            Vector ret = (Vector)softmaxFunc((Vector)_neuronActivations);
            for (int i = 0; i < x.Length; i++)
            {
                // ret[i] = the derivative of Si w.r.t Zi
                ret[i] *= 1-ret[i];
            }
            return ret;
        }



        #endregion
        
    }
}

namespace ML
{
    public class ReLULayer:ActionLayer
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
            return new Scalar(Math.Max(x.Data, 0));
        }

        private static Tensor reluDeriv(Tensor x)
        {
            float ret = 0;
            // the relu derivative: 
            //{ x > 0 : 1}
            //{ x <= 0: 0}
            if (x.Data > 0)
                ret = 1;
            return new Scalar(ret);
        }

        #endregion Methods

    }
}

namespace ML
{
    public class LinearLayer:ActionLayer
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
            return new Scalar(x.Data);
        }

        private static Tensor linearDeriv(Tensor x)
        {
            return new Scalar(1);
        }

        #endregion Methods

    }
}