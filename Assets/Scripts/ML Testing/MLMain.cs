using System;
using System.Collections;
using System.Collections.Generic;
using ML;
using UnityEngine;
using Network = ML.Network;
using Random = System.Random;

public class MLMain : MonoBehaviour
{
    private static int sampleSize = 100000;
    private static int batchSize = 250;

    private float noise = 0.00001f;
    
    
    // counts loops through update
    private int counter = 0;
    // log every LOG_INTERVAL loops through update
    private int LOG_INTERVAL = 1;
    
    private float lr = 0.00001f;
    private Tensor[] features = new Tensor[sampleSize];
    private Tensor[] labels = new Tensor[sampleSize];
    private Random rand = new Random();
    private double x ;

    // creating the softmax activation func 
    //private ML.Function<Tensor,Tensor> softmax = new Function<Tensor,Tensor>( softmaxFunc,softmaxDeriv,"softmax");
    // creating the mse loss
    private ML.Loss mse = new Loss( mseFunc,mseDeriv,"mse");
    // creating the Categorical crossEntropy loss
    private ML.Loss CE = new Loss( CEFunc,CEDeriv,"categorical crossentropy");
    // creating the Binary Catergorical crossEntropy loss
    private ML.Loss BCE = new Loss( BCCEFunc,CEDeriv,"binary categorical crossentropy");
    private ML.Loss SoftMaxCE = new Loss( SoftMaxCEFunc,SoftMaxCEDeriv,"softmax categorical crossentropy");
    private Network net;
    
    
    // Start is called before the first frame update
    void Start()
    {
        x = rand.NextDouble() * 10;
        // generating the data
        Debug.Log("Generating data...");
        
        for (int i = 0; i < sampleSize; i++)
        {
            features[i] = new Tensor(1, x,"Data "+i);
            labels[i] = new Tensor(2,x*x,  "Label " + i);
            labels[i][0].Value = 0;
            if (x > 5)
                labels[i][0].Value = 1;
            labels[i][1].Value = 1 - labels[i][0].Value;
            x = rand.NextDouble() * 10;

        }

        Debug.Log("Done!");
        Layer[] layers=
        {
            new DenseLayer(8,"selu","d4"),
            new DenseLayer(8,"selu","d4"),
            new DenseLayer(labels[0].Length,"softmax","d4",useLeCun:true),
        };
        Optimizer optimizer = new SGDMomentum(batchSize,0.95);
        net = new Network(layers,lr,1,BCE,optimizer);

    }
    #region activations
    
    private static Tensor mseFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(1);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[0].Value += Math.Pow(input.pred[i].Value - input.label[i].Value,2);
        }

        return ret;
    }

    private static Tensor mseDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i].Value = 2*(input.pred[i].Value - input.label[i].Value);
        }
        return ret;

    }

    private static Tensor CEFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(1);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[0].Value += -input.label[i].Value * Math.Log(input.pred[i].Value);
        }
        return ret;
    }

    private static Tensor CEDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        
        Tensor ret = new Tensor(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i].Value = -(input.label[i].Value / input.pred[i].Value);
        }
        return ret;

    }
    private static Tensor BCCEFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(1,"BCE Loss");
        //L = -t1*log(s1) - (1-t1)*log(1-s1)
        ret[0].Value = -input.label[0].Value * Math.Log(input.pred[0].Value) - input.label[1].Value * Math.Log(input.pred[1].Value);
        return ret;
    }

    private static Tensor BCCEDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(input.pred.Length,"BCCEDeriv");
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i].Value = -(input.label[i].Value / input.pred[i].Value) + (1 - input.label[i].Value)/(1-input.pred[i].Value);
        }
        return ret;

    }


    public static Tensor SoftMaxCEFunc((Tensor pred, Tensor label) input)
    {
        input.pred = softmax(input.pred);
        Tensor ret = new Tensor(1,"softmax ce loss");
        
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[0].Value += -input.label[i].Value * Math.Log(input.pred[i].Value);
        }
        return ret;
    }
    public static Tensor SoftMaxCEDeriv((Tensor pred, Tensor label) input)
    {
        input.pred = softmax(input.pred);
        Tensor ret = new Tensor(input.pred.Length);
        
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i].Value = input.pred[i].Value - input.label[i].Value;
        }

        return ret;
    }

    public static Tensor softmax(Tensor x)
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
            ret[i].Value = ret[i].Value;// check if the recursive call works!!
        }

        return ret;
    }
    
    #endregion
    
    // a method that does the logging. yeah it's a great comment.
    void log()
    {
        Debug.Log(net.ToString());
    }
    
    
    // Update is called once per frame
    void Update()
    {
        net.backwards(features,labels);
        counter++;
        if (counter % LOG_INTERVAL == 0)
        {
            log();
        }
        
    }
}
