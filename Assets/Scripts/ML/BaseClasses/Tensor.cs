using System;
using System.Collections;
using System.Collections.Generic;
using ML;
using UnityEngine;
using UnityEngine.Assertions;

public  abstract class Tensor
{
    #region variables
    // every tensor needs dimensionality 
    private int _dimension;
    
    public int Dimension
    {
        get => _dimension;
        set => _dimension = value;
    }
    
    // name, doesn't need to have it
    private string _name = "";

    public string Name
    {
        get => _name;
        set => _name = value;
    }
    
    // every tensor needs to have data, but it might be overridden in inheritance.
    // we set data to a float array because if its a scalar then it's just easier to handle everything with array.
    protected float[] _data = {0};
    public  float Data
    {
        get => _data[0];
        set => _data[0] = value;
    }
    public  float[] DataArr
    {
        get => _data;
        set => _data = value;
    }
    
    // length is the size of the data
    private int _length = 1;
    public int Length
    {
        get => _length;
        protected set => _length = value;
    }

    
    #endregion

    #region constructors
    // constructor with name
    protected Tensor(int dimension, string name="")
    {
        _dimension = dimension;
        _name = name;
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

    // square brackets operator
    public virtual float this[int i]
    {
        get { return _data[i]; }
        set { Data = value; }
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

    public abstract Tensor ElementWiseFunction(Func<Tensor, Tensor> func);


    public abstract Tensor ElementWiseMultiply(Tensor a);
    
    public abstract Tensor Clone();
    #endregion

}
