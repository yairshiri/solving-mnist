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
}