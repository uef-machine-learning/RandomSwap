# RandomSwap
Implementation of the Random Swap (RLS-2) [1] algorithm.

# Pseudo-Code
```
Data structures: 
----------------- 

N       number of data objects
k       number of clusters
D       number of attributes
T       number of iterations
X[N]    array of N D-dimensional vectors; data objects (feature vectors)
P[N]    array of N integers pointing from X to C; partition
C[k]    array of k D-dimensional vectors; cluster representatives (centroids)

Constants N, k, D and T are global.



Algorithm: 
----------- 

In every function all parameters are value parameters.


PerformRS(X) {
    /* initial solution */
    C := SelectRandomRepresentatives(X);
    P := OptimalPartition(C,X);
    
    FOR i := 1 TO T DO 
        {
        (C',j)  := RandomSwap(C,X);
        P'      := LocalRepartition(P,C',X,j);
        (P',C') := K-means(P',C',X);
        
        IF ObjectiveFunction(P',C',X) < ObjectiveFunction(P,C,X) THEN 
           {
           (P,C) := (P',C');
           }
        }
    
    RETURN (P,C);
}


K-means(P,C,X) {
    /* performs two K-means iterations */
    FOR i := 1 TO 2 DO 
        {
        /* OptimalRepresentatives-operation should be before 
        OptimalPartition-operation, because we have previously tuned 
        partition with LocalRepartition-operation */         
        C := OptimalRepresentatives(P,X);    
        P := OptimalPartition(C,X);
        }
    
    RETURN (P,C);
}


OptimalPartition(C,X) {
    FOR i := 1 TO N DO 
        {
        P[i] := FindNearestRepresentative(C,X[i]);
        }
    
    RETURN P;
}


OptimalRepresentatives(P,X) {
    /* initialize Sum[1..k] and Count[1..k] by zero values! */

    /* sum vector and count for each partition */
    FOR i := 1 TO N DO 
        {
        j := P[i];
        Sum[j] := Sum[j] + X[i];
        Count[j] := Count[j] + 1;
        }
    
    /* optimal representatives are average vectors */
    FOR i := 1 TO k DO 
        {
        IF Count[i] <> 0 THEN 
           {
           C[i] := Sum[i] / Count[i];
           }        
        }    
        
    RETURN C;
}


FindNearestRepresentative(C,x) {
    j := 1;
    
    FOR i := 2 TO k DO 
        {
        IF Dist(x,C[i]) < Dist(x,C[j]) THEN 
           {
           j := i;
           }
        }
    
    RETURN j;
}


SelectRandomRepresentatives(X) {
    FOR i := 1 TO k DO  
        {
        C[i] := SelectRandomDataObject(C,X,i-1);
        }
    
    RETURN C;
}


SelectRandomDataObject(C,X,m) {
    REPEAT {
        i := Random(1,N);   
        ok := True
        
        /* eliminate duplicates */
        FOR j := 1 TO m DO 
            {
            IF C[j] = X[i] THEN 
                {
                ok := False;
                }
            }
        
    } UNTIL ok = True;
    
    RETURN X[i];
}


RandomSwap(C,X) {
    j := Random(1,k);
    C[j] := SelectRandomDataObject(C,X,k);
    
    RETURN (C,j);
}


LocalRepartition(P,C,X,j) {
    /* object rejection */
    FOR i := 1 TO N DO 
        {
        IF P[i] = j THEN 
            {
            P[i] := FindNearestRepresentative(C,X[i]);
            }
        }
    
    /* object attraction */
    FOR i := 1 TO N DO 
        {
        IF Dist(X[i],C[j]) < Dist(X[i],C[P[i]]) THEN 
            {
            P[i] := j;
            }
        }
    
    RETURN P;
}


/* this (example) objective function is sum of squared distances
   of the data object to their cluster representatives */
ObjectiveFunction(P,C,X) {
    sum := 0;
    
    FOR i := 1 TO N DO 
        {
        /* distance to the power of two */
        sum := sum + Dist(X[i],C[P[i]])^2;
        }
    
    RETURN sum;
}



Dist(x1,x2)     calculates euclidean distance between vectors x1 and x2
Random(a,b)     returns random number between a..b
```

# References
[1] Pasi Fränti and Juha Kivijärvi. "Randomized local search algorithm for the clustering problem". Pattern Analysis and Applications, 3 (4), 358-369, 2000. https://link.springer.com/article/10.1186/s40537-018-0122-y
    
