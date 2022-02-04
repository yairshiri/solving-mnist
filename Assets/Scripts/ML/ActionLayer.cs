using System;
using UnityEngine.Assertions;

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