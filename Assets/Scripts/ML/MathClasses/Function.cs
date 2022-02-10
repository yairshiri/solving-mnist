using System;

namespace ML
{
    public class Function<T,TReasult>
    {
        #region variables

        private string _name;
        private Func<T, TReasult> _function;
        private Func<T, TReasult> _deriv;

        public Func<T, TReasult> Func
        {
            get => _function;
            set => _function = value;
        }

        public Func<T, TReasult> FunctionDeriv
        {
            get => _deriv;
            set => _deriv = value;
        }
        public string Name
        {
            get => _name;
            set => _name = value;
        }


        #endregion

        #region constructors
        // a function has to have a function, and a derivative. doesn't need to have name
        public Function(Func<T,TReasult> function,Func<T,TReasult> deriv,string name)
        {
            Func = function;
            FunctionDeriv = deriv;
            Name = name;
        }
        public Function(Func<T,TReasult> function,Func<T,TReasult> deriv)
        {
            Func = function;
            FunctionDeriv = deriv;
        }

        #endregion
    }
    
    // activation get a tensor and return a tensor
    public class Activation:Function<float,float>{
        public Activation(Func<float, float> function, Func<float, float> deriv, string name) : base(function, deriv, name)
        {
        }

        public Activation(Func<float, float> function, Func<float, float> deriv) : base(function, deriv)
        {
        }
    }
    
    // loss function revive two tensors and return a tensor
    public class Loss:Function<(Tensor, Tensor),Tensor>
    {
        public Loss(Func<(Tensor, Tensor), Tensor> function, Func<(Tensor, Tensor), Tensor> deriv, string name) : base(function, deriv, name)
        {
        }

        public Loss(Func<(Tensor, Tensor), Tensor> function, Func<(Tensor, Tensor), Tensor> deriv) : base(function, deriv)
        {
        }
    }

    
}