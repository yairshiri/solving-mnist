using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using ML;
using UnityEngine;
using UnityEngine.Assertions;

public   class Tensor
{
    #region variables
    // every tensor needs dimensionality 
    private int[] _shape;
    // only the final tensor has a value
    private double _value = double.NaN;
    // name, doesn't need to have it
    private string _name = "";
    // need to add explanation here
    protected Tensor[] _data ;
    
    #endregion

    #region GettersAndSetters
    public string Name
    {
        get => _name;
        set => _name = value;
    }

    public int[] Shape
    {
        get => _shape;
        set => _shape = value;
    }
    
    public int Dimension
    {
        get => _shape.Length;
    }
    public  Tensor[] Data
    {
        get => _data;
        set => _data = value;
    }

    public double Value
    {
        get => _value;
        set => _value = value;
    }

    #endregion
    #region constructors
    // constructor with name
    public Tensor(int[] shape, string name="")
    {
        Shape = shape;
        Name = name;
        Data = new Tensor[Shape[0]];
        // if we are a scalar or vector. 
        if (Dimension == 1)
        {
            for (int i = 0; i < shape[0]; i++)
            {
                Data[i] = new Tensor(0,  name+"["+i+"]");
            }
        }
        for (int i = 0; i < shape[0]; i++)
        {
            Data[i] = new Tensor(Shape.Skip(1).ToArray(), name+"["+i+"]");
        }
    }

    public Tensor(double value, string name = "")
    {
        Shape = new[] {1};
        Value = value;
        Name = name;
    }
    // copying constructor
    public Tensor(Tensor a)
    {
        Name = a.Name;
        Shape = a.Shape;
        Data = new Tensor[Shape[0]];
        if (Dimension == 1)
        {
            for (int i = 0; i < Shape[0]; i++)
            {
                Data[i] = new Tensor(a[i].Value,  Name+"["+i+"]");
            }
        }
        for (int i = 0; i < Shape[0]; i++)
        {
            Data[i] = new Tensor(a[i]);
        }

    }
    #endregion

    #region operators
    // trying to make a general + operator for tensors
    /*public static Tensor operator +(Tensor a, Tensor b)
    {
        // if at least one of the tensors is a scalar, just add the scalar to all elements. 
        // we do it with the Math.Min(length,i) because if the length is 1, then it means that
        // it's a scalar (or a vector/matrix with size 1) and we can iterate over the data with i.
        if (a.Dimension != 0 && b.Dimension != 0)
        {
            // if a and b are not scalars, check dimensionality
            Assert.AreEqual(a._dimension,b._dimension);
            
        }
        else
        {
            // checking for the same size if there is not scalar
            Assert.AreEqual(a.Length,b.Length);
        }
        // creating the return value after all of the checks
        Tensor ret = new Tensor(Math.Max(a._dimension,b._dimension));

        // the addition elements wise. the Math.Min is explained above.
        for (int i = 0; i < ret._length; i++)
        {
            ret[i] = a[Math.Min(a.Length-1,i)] + b[Math.Min(b.Length-1,i)];
        }
        
        return ret;
    }*/

    public static Tensor operator *(Tensor a, Tensor b)
    {
        if (Math.Abs(a.Dimension - b.Dimension) > 1)
            throw new Exception("The dimentions of a and b need to be equal or 1 less of each other!");
        Tensor max;
        Tensor min;
        if (a.Dimension > b.Dimension)
        {
            max = a;
            min = b;
        }
        else
        {
            max = b;
            min = a;
        }
        Tensor ret = new Tensor(max.Shape);
        // i is the max index, j is min.
        
        return ret;
    }

    // square brackets operator
    public virtual Tensor this[int i]
    {
        get => Data[i];
        set => Data[i] = value;
    }
    
    

    #endregion

    #region methods
    
    public new virtual  string ToString()
    {
        // all tensors may have names, so instead of implementing this (the name adding) for every claas we implement here
        // and use in other classes (with base.ToString())
        string ret = "";
        if (Name != "")
            ret += Name+": ";
        return ret;
    }

    public virtual Tensor ElementWiseFunction(Func<Tensor, Tensor> func)
    {
        throw new NotImplementedException();
    }


    public virtual Tensor ElementWiseMultiply(Tensor a)
    {
        throw new NotImplementedException();
    }
    
    public virtual Tensor Clone()
    {
        throw new NotImplementedException();
    }

    public virtual Tensor Transpose()
    {
        throw new NotImplementedException();
    }

    public virtual bool Multiplyable(Tensor a)
    {
        throw new NotImplementedException();
    }
    #endregion

}
