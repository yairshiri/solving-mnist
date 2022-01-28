using System;
using System.Collections;
using System.Collections.Generic;
using ML;
using UnityEngine;
using Network = ML.Network;
using Vector = ML.Vector;
using Layer = ML.DenseLayer;
using Random = System.Random;

public class MLMain : MonoBehaviour
{
    private int sampleSize = 100000;

    private float noise = 0.00001f;
    
    
    // counts loops through update
    private int counter = 0;
    // log every LOG_INTERVAL loops through update
    private int LOG_INTERVAL = 20;
    
    private float lr = 0.0001f;
    private Vector features = new Vector(1);
    private Vector labels = new Vector(2);
    private Random rand = new Random();
    private float x ;

    // creating the relu activation func 
    private ML.Function<float, float> relu = new Function<float, float>( reluFunc,reluDeriv,"relu");
    // creating the relu activation func 
    private ML.Function<float, float> sigmoid = new Function<float, float>( sigFunc,sigDeriv,"sigmoid");
    // creating the linear activation func 
    private ML.Function<float, float> linear = new Function<float, float>( linearFunc,linearDeriv,"linear");
    // creating the softmax activation func 
    private ML.Function<Vector,Vector> softmax = new Function<Vector,Vector>( softmaxFunc,softmaxDeriv,"softmax");
    // creating the mse loss
    private ML.Function<(Vector,Vector), Vector> mse = new Function<(Vector,Vector), Vector>( mseFunc,mseDeriv,"mse");

    private Network net;
    
    
    // Start is called before the first frame update
    void Start()
    {
        Layer[] layers=
        {
            
            new ML.DenseLayer(3,relu,"d1"),
            new ML.DenseLayer(4,relu,"d2"),
            new ML.DenseLayer(2,linear,"output")
        };
        net = new Network(layers,lr,1,mse);
        x = (float)rand.NextDouble() * 10;
    }

    private static float reluFunc(float x)
    {
        return Math.Max(x, 0);
    }

    private static float sigFunc(float x)
    {
        return 1.0f / (float)(1 + Math.Exp(-x));
    }
    private static float sigDeriv(float x)
    {
        return sigFunc(x) * (1-sigFunc(x));
    }

    private static float reluDeriv(float x)
    {
        return Math.Max(x, 0)/Math.Abs(x);
    }
    private static float linearFunc(float x)
    {
        return x;
    }

    private static float linearDeriv(float x)
    {
        return 1;
    }

    private static Vector softmaxFunc(Vector x) 
    {
        // initializing a new vector with the length of x.
        Vector ret = new Vector(x.Length);
        // every element is e to the power of the elements devided by the sum of e to the poewr of all the elements.
        // implementation from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        float sum = 0;
        // finding the max value in x
        float max = 0;
        for (int i = 0; i < ret.Length; i++)
            if (x[i] > max)
                max = x[i];
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
        }
        return ret;
    }
    // softmax deriv from https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    private static Vector softmaxDeriv(Vector x)
    {
        Vector ret = softmaxFunc(x);
        for (int i = 0; i < x.Length; i++)
        {
            ret[i] = ret[i] * (1 - ret[i]);
        }
        return ret;
    }
    

    private static Vector mseFunc((Vector x, Vector y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        Vector ret = new Vector(1);
        for (int i = 0; i < input.x.Length; i++)
        {
            ret[0] += (float)Math.Pow(input.x[i] - input.y[i],2);
        }

        return ret;
    }

    private static Vector mseDeriv((Vector x, Vector y) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.x.Length==input.y.Length);
        Vector ret = new Vector(input.x.Length);
        for (int i = 0; i < input.x.Length; i++)
        {
            ret[i] = 2*(input.x[i] - input.y[i]);
        }
        return ret;

    }
    
    // a method that does the logging
    void log()
    {
        Debug.Log(net.ToString());
    }
    
    
    // Update is called once per frame
    void Update()
    {
        x = (float)rand.NextDouble() * 10;
        features[0] = x;
        if (x > 5)
        {
            labels[0] = 0;
            labels[1] = 1;
        }
        else
        {
            labels[0] = 1;
            labels[1] = 0;
        }
        net.backwards(features,labels);
        counter++;
        if (counter % LOG_INTERVAL == 0)
        {
            log();
        }
        
    }
}
