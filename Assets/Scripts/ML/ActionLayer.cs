using System;

namespace ML
{
    public class ActionLayer:Layer
    {
        #region Fields
        private Function<Tensor,Tensor> action;
        #endregion Fields
        #region Constructors
        // Action layers (such as flatten) have no activation function
        public ActionLayer(int[] shape,Function<Tensor,Tensor> action, string name="") : base(shape,name)
        {
            this.action = action;
        }
        #endregion
        
        public override Tensor Forwards(Tensor input)
        {
            return action.Func(input);
        }

        public override (Tensor, Matrix, Vector) Backwards(Tensor input)
        {
            return (action.FunctionDeriv(input), null, null);
        }

        public override void ApplyGradients(Matrix wGrads, Vector bGrads)
        {
            // no learnable parameters to learn so nothing to see here -_-
        }
    }
}