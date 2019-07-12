/*-------------------------------------------------------------------*/
/* RS.C         Marko Tuononen, Pasi Fränti                          */
/*                                                                   */
/* Advanced model implementation of Random Swap (RS)                 */
/*                                                                   */
/*   "Randomized local search algorithm for the clustering problem"  */
/*   Pattern Analysis and Applications, 3 (4), 358-369, 2000.        */
/*   Pasi Fränti and Juha Kivijärvi                                  */
/*                                                                   */
/* This model also includes possibility to select deterministicly,   */
/* which cluster centroid to swap. In case of deterministic choice   */
/* in every iteration cluster that increases objective function      */
/* (MSE) least, if removed, is selected.                             */
/*                                                                   */
/* K-means -operation uses activity detection presented in           */
/*                                                                   */
/*   "A fast exact GLA based on code vector activity detection"      */
/*   IEEE Trans. on Image Processing, 9 (8), 1337-1342, August 2000. */
/*   Timo Kaukoranta, Pasi Fränti and Olli Nevalainen                */
/*                                                                   */
/* Naming conventions used in the code                               */
/*                                                                   */
/*    TS        training set (data objects)                          */
/*    CB        codebook (cluster representatives, centroids)        */
/*    P         partitioning (pointing from TS to CB)                */
/*                                                                   */
/*    p-prefix  pointer, e.g. pTS is pointer to the training set TS  */
/*                                                                   */
/* ----------------------------------------------------------------- */
/*                                                                   */
/* Traveller search mode implementation is based on idea presented   */
/* in following paper:                                               */
/*                                                                   */
/*  "Faster and more robust point symmetry-based K-means algorithm"  */
/*  Pattern Recognition, 40, 410-422, 2007.                          */
/*  Kuo-Liang Chung, Jhin-Sian Lin                                   */
/*                                                                   */
/* The main idea of the mode is that the optimal centroid search for */
/* the closer data points is filtered to include only the current    */
/* centroid and all active centroids that have moved greater distance*/
/* and the current centroid.					     */
/*                                                                   */
/* ----------------------------------------------------------------- */
/*                                                                   */
/* HISTORY:                                                          */
/*                                                                   */
/* 0.35 PF  Added: RandomCodebook, LuxburgInitialization (21.2.17)   */
/* 0.33 PF  MonitorProgress mode + Stochastic variant (8.7.16)       */
/* 0.31 PF  Renamed RLS->RS; Refactoring code; CI-index (25.6.16)    */
/* 0.25 AH  Traveller search and new Q-level 4 features (28.2.10)    */
/* 0.24 MM  Correct random initialization (15.7.09)                  */
/* 0.22 VH  Correct Random initialization (15.4.09)                  */
/* 0.21 MT  Modified RSInfo() to print less information              */
/* 0.20 MT  Fixed SelectRandomDataObject, added automatic iter.count */
/*-------------------------------------------------------------------*/


#define ProgName       "RS"
#define VersionNumber  "Version 0.35"
#define LastUpdated    "20.9.2016"  /* PF */

/* converts ObjectiveFunction values to MSE values */
#define CALC_MSE(val) (double) (val) / (TotalFreq(pTS) * VectorSize(pTS))

#define AUTOMATIC_MAX_ITER  50000
#define AUTOMATIC_MIN_SPEED 1e-5
#define min(a,b) ((a) < (b) ? (a) : (b))

/*-------------------------------------------------------------------*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <string.h>

#include "cb.h"
#include "random.h"
#include "interfc.h"
#include "reporting.h"


/* ========================== PROTOTYPES ============================= */

int PerformRS(TRAININGSET *pTS, CODEBOOK *pCB, PARTITIONING *pP, int iter,
    int kmIter, int deterministic, int travellerSearch, int quietLevel,
    int useInitialCB, int monitoring);
void InitializeSolution(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS,
    int clus);
void FreeSolution(PARTITIONING *pP, CODEBOOK *pCB);
YESNO StopCondition(double currError, double newError, int iter);
llong GenerateInitialSolution(PARTITIONING *pP, CODEBOOK *pCB,
    TRAININGSET *pTS, int useInitialCB);
void SelectRandomRepresentatives(TRAININGSET *pTS, CODEBOOK *pCB);
void SelectRandomRepresentatives2(TRAININGSET *pTS, CODEBOOK *pCB); /* mm */
int SelectRandomDataObject(CODEBOOK *pCB, TRAININGSET *pTS);
void RandomCodebook(TRAININGSET *pTS, CODEBOOK *pCB);
void RandomSwap(CODEBOOK *pCB, TRAININGSET *pTS, int *j, int deterministic, 
    int quietLevel);
void LocalRepartition(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS, 
    int j, double time, int quietLevel);
void OptimalRepresentatives(PARTITIONING *pP, TRAININGSET *pTS,
    CODEBOOK *pCB, int *active, llong *cdist, int *activeCount, int travellerSearch);
int BinarySearch(int *arr, int size, int key);
void OptimalPartition(CODEBOOK *pCB, TRAININGSET *pTS, PARTITIONING *pP,
    int *active, llong *cdist, int activeCount, llong *distance, int quietLevel);
void KMeans(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS,
    llong *distance, int iter, int travellerSearch, int quietLevel, double time);
llong ObjectiveFunction(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS);
void CalculateDistances(TRAININGSET *pTS, CODEBOOK *pCB, PARTITIONING *pP,
    llong *distance);
int FindSecondNearestVector(BOOKNODE *node, CODEBOOK *pCB, int firstIndex,
    llong *secondError);
int SelectClusterToBeSwapped(TRAININGSET *pTS, CODEBOOK *pCB, 
    PARTITIONING *pP, llong *distance);
char* RSInfo(void);


/* ========================== FUNCTIONS ============================== */

/* Gets training set pTS (and optionally initial codebook pCB or 
   partitioning pP) as a parameter, generates solution (codebook pCB + 
   partitioning pP) and returns 0 if clustering completed successfully. 
   N.B. Random number generator (in random.c) must be initialized! */

int PerformRS(TRAININGSET *pTS, CODEBOOK *pCB, PARTITIONING *pP, int iter, 
int kmIter, int deterministic, int travellerSearch, int quietLevel, 
int useInitial, int monitoring)
{
  PARTITIONING  Pnew;
  CODEBOOK      CBnew, CBref;
  int           i, j, better;
  int           ci=0, ciPrev=0, ciZero=0, ciMax=0, PrevSuccess=0;
  int           CIHistogram[111];
  llong         currError, newError;
  llong         distance[BookSize(pTS)];
  double        c, error;
  int           stop=NO, automatic=((iter==0) ? YES : NO);
  
  /* Error checking for invalid parameters */ 
  if ((iter < 0) || (kmIter < 0) || (BookSize(pTS) < BookSize(pCB)))
    {
    return 1;  // Error: clustering failed
    }

  /* Progress monitor uses input codebook as reference */
  if (monitoring)
    {
    CreateNewCodebook(&CBref, BookSize(pCB), pTS);
    CopyCodebook(pCB, &CBref);
    for( ci=0; ci<=100; ci++ ) CIHistogram[ci]=0;
    useInitial *= 100;  /* Special code: 0->0, 1->100, 2->200 */
    }
  InitializeSolution(&Pnew, &CBnew, pTS, BookSize(pCB));
  SetClock(&c);
  currError = GenerateInitialSolution(pP, pCB, pTS, useInitial);
  error = CALC_MSE(currError);
  if(useInitial) ciPrev = CentroidIndex(&CBref, pCB);
  else           ciPrev = 100;
  
  /* use automatic iteration count */
  if (automatic)  iter = AUTOMATIC_MAX_ITER;
  
  PrintHeader(quietLevel);
  PrintIterationRS(quietLevel, 0, error, 0, GetClock(c), 1);

  /* Deterministic variant initialization */
  if (deterministic)
    {
    CalculateDistances(pTS, pCB, pP, distance);
    j = SelectClusterToBeSwapped(pTS, pCB, pP, distance);
    }
  
  /* - - - - -  Random Swap iterations - - - - - */

  for (i=1; (i<=iter) && (!stop); i++)
    {
    better = NO;

    /* generate new solution */
    CopyCodebook(pCB, &CBnew);
    CopyPartitioning(pP, &Pnew);
    RandomSwap(&CBnew, pTS, &j, deterministic, quietLevel);

    /* tuning new solution */
    LocalRepartition(&Pnew, &CBnew, pTS, j, c, quietLevel);
    KMeans(&Pnew, &CBnew, pTS, distance, kmIter, travellerSearch, quietLevel, c);

    newError = ObjectiveFunction(&Pnew, &CBnew, pTS);
    error    = CALC_MSE(newError);

    /* Found better solution */
    if (newError < currError)
      {
      /* Monitoring outputs CI-value: relative to Prev or Reference */
      if(monitoring)  
         {
         if(useInitial) ci = CentroidIndex(&CBnew, &CBref);
         else           ci = CentroidIndex(&CBnew, pCB);
         /* CI decreases: update Success histogram */
         if( (ci>=0) && (ci<ciPrev) && (ci<100) )
           {
           /* printf("XXXX Prev=%d  Curr=%d  CI=%d  CIPrev=%i  Iter=%d \n", PrevSuccess, i, ci, ciPrev, (i-PrevSuccess)); */
           CIHistogram[ci] += (i-PrevSuccess);
           if(ci>ciMax)  ciMax = ci;
           if(ci==0)     ciZero = i;
           PrevSuccess = i;
           }
         /* CI increases: report warning message */
         if( (ci>ciPrev) && (quietLevel) ) printf("!!! CI increased %i to %i at iteration %d\n", ciPrev, ci, i);
         /* Remember to update CI value */
         ciPrev = ci;
         /* If monitoring, then stop criterion is CI=0 */
         if(automatic) stop=(ci==0); 
         }

      /* Check stopping criterion */
      else if(automatic)
        {
        stop = StopCondition(currError, newError, i);
        }

      CopyCodebook(&CBnew, pCB);
      CopyPartitioning(&Pnew, pP);
     
      currError = newError;
      better = YES;
      
      if (deterministic) /* Alterantive ro Random. But why here?  */
        {
        j = SelectClusterToBeSwapped(pTS, pCB, pP, distance);
        }
      }

    PrintIterationRS(quietLevel, i, error, ci, GetClock(c), better);
    }

  /* - - - - -  Random Swap iterations - - - - - */

  error = CALC_MSE(currError);  
  PrintFooterRS(quietLevel, i-1, error, GetClock(c));

  if(monitoring && quietLevel)  
     {
     PrintMessage("Total: %-7d   Swaps: ", ciZero);
     for( ci=0; ci<=ciMax; ci++ )
       {
       PrintMessage("%3d  ", CIHistogram[ci]);
       }
     PrintMessage("\n", ciZero);
     }

  FreeSolution(&Pnew, &CBnew);
  return 0;
}  


/*-------------------------------------------------------------------*/


void InitializeSolution(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS, 
int clus)
{
  CreateNewCodebook(pCB, clus, pTS);
  CreateNewPartitioning(pP, pTS, clus);
} 


/*-------------------------------------------------------------------*/


void FreeSolution(PARTITIONING *pP, CODEBOOK *pCB)
{
  FreeCodebook(pCB);
  FreePartitioning(pP);
} 


/*-------------------------------------------------------------------*/

            
YESNO  StopCondition(double currError, double newError, int iter)
{
  static double   currImpr=DBL_MAX, prevImpr=DBL_MAX;
  static int      prevIter=1;

  currImpr  = (double)(currError - newError) / (double)currError;
  currImpr /= (double) (iter - prevIter);
  if (AUTOMATIC_MIN_SPEED < currImpr + prevImpr)
     {
     prevImpr = currImpr;
     prevIter = iter;
     return(NO);
     }
  else  /* too slow speed, better to stop.. */
     {
     return(YES);
     }
}


/*-------------------------------------------------------------------*/


llong GenerateInitialSolution(PARTITIONING *pP, CODEBOOK *pCB, 
TRAININGSET *pTS, int useInitial)
{
  if (useInitial == 1)
  {
    GenerateOptimalPartitioningGeneral(pTS, pCB, pP, MSE);
  } 
  else if (useInitial == 2)
  {
    GenerateOptimalCodebookGeneral(pTS, pCB, pP, MSE);
  } 
  else
  {
    SelectRandomRepresentatives(pTS, pCB);
    GenerateOptimalPartitioningGeneral(pTS, pCB, pP, MSE);
  }

  return ObjectiveFunction(pP, pCB, pTS);
}


/*-------------------------------------------------------------------*/
// Copied from Ismo's GenerateRandomCodebook() from cb_util.c
/*-------------------------------------------------------------------*/


void SelectRandomRepresentatives(TRAININGSET *pTS, CODEBOOK *pCB)
{
  
  int k, n, x, Unique;

  for (k = 0; k < BookSize(pCB); k++) 
    {
    do 
      {
      Unique = 1;
      x = irand(0, BookSize(pTS) - 1);
      for (n = 0; (n < k) && Unique; n++) 
         Unique = !EqualVectors(Vector(pTS, x), Vector(pCB, n), VectorSize(pCB));
      } 
    while (!Unique);

    CopyVector(Vector(pTS, x), Vector(pCB, k), VectorSize(pCB));
    VectorFreq(pCB, k) = 0;
    }

}


/*-------------------------------------------------------------------*/
/* Initializes codebook with randomly selected vectors from dataset. */
/* Duplicate vectors are allowed.                                    */


void SelectRandomRepresentatives2(TRAININGSET *pTS, CODEBOOK *pCB) /* mm */
{
  int i,j;

  TRAININGSET tempTS;
  CreateNewCodebook(&tempTS,BookSize(pTS),pTS);
  CopyCodebook(pTS,&tempTS);
  for (i=0;i<BookSize(pCB);i++)
    {
    j = IRI(i,BookSize(&tempTS));
    CopyVector(Vector(&tempTS,j),Vector(pCB,i),VectorSize(&tempTS));
    VectorFreq(pCB,i)=1;
    CopyVector(Vector(&tempTS,i),Vector(&tempTS,j),VectorSize(&tempTS));
    }
  FreeCodebook(&tempTS);
}


/*-------------------------------------------------------------------*/
/* Pasi's solution 20.9.2016                                         */
/* (1) Shuffle training set. (2) Select first k vectors.             */
/* Someone else please test this...                                  */
/*-------------------------------------------------------------------*/


void RandomCodebook(TRAININGSET *pTS, CODEBOOK *pCB)
{
  int i;

  ShuffleTS(pTS);
  for(i=0; i<BookSize(pCB); i++) 
     {
     CopyNode( &Node(pTS,i), &Node(pCB,i), VectorSize(pCB));
     }       
}


/*-------------------------------------------------------------------*/
/* Marko's variant: uniform sampling. Not random if TS sorted!       */
/*-------------------------------------------------------------------*/


void SelectRandomRepresentativesbyMarko(TRAININGSET *pTS, CODEBOOK *pCB)
{
  int i, j, k;

  k = BookSize(pTS) / BookSize(pCB);

  for (i = 0; i < BookSize(pCB); i++) 
    {
      /* interval [0,BookSize(pTS)) is divided M subintervals and 
	 random number is chosen from every subinterval */
      if (i == (BookSize(pCB) - 1))
	{   
	  /* random number generator must be initialized! */
	  j = IRI(i*k, BookSize(pTS));
	} 
      else 
	{   
	  j = IRI(i*k, (i+1)*k);
	}

      CopyVector(Vector(pTS, j), Vector(pCB, i), VectorSize(pTS));
      VectorFreq(pCB, i) = 1;
    }

}


/*-------------------------------------------------------------------*/
/*  Simple vector by Ville  */ 
/*-------------------------------------------------------------------*/


llong* GetLongVector(dim)
{
  llong *ptr; 

  ptr = (llong *) malloc(sizeof(llong)*dim); 
    
  if (ptr == NULL) {
    printf("Out of memory: in GetLongVector()\n"); 
    exit(-1); 
  }
    
  return ptr;
}


/*-------------------------------------------------------------------*/
/* Random selection of type k-means++:                               */
/* The Advantages of Careful Seeding by Arthur and Vassilvitskii     */
/* Code by Ville Hautamäki (following original C++ implementation)   */
/*-------------------------------------------------------------------*/


void SelectRandomWeightedRepresentatives(TRAININGSET *pTS, CODEBOOK *pCB)
{
  int i, index, n, numCenters, centerCount, localTrial, numLocalTries = 1;
  llong *vector, currentPot, randval, newPot,bestNewPot; /* temprand */

  n = BookSize(pTS);
  numCenters = BookSize(pCB); 
  index = irand(0, n - 1);
  CopyVector(Vector(pTS, index), Vector(pCB, 0), VectorSize(pCB));
  vector = GetLongVector(n);  

  currentPot = 0; 

  for (i=0; i<n-1; i++) 
    {
    vector[i] = VectorDist(Vector(pTS, i), Vector(pCB, 0), VectorSize(pCB));
    currentPot += vector[i];
    }

  for (centerCount = 1; centerCount < numCenters; centerCount++) 
    {
    bestNewPot = -1;
    for (localTrial = 0; localTrial < numLocalTries; localTrial++) 
      {
      randval = (llong) (frand() * (float) currentPot);
      /*  temprand = randval; */
      for (index = 0; index < n; index++) 
        {
	if (randval <= vector[index])  break;
	else                           randval -= vector[index];
        }
      newPot = 0; 
      for (i= 0; i < n; i++) 
	newPot += min(VectorDist(Vector(pTS, i), 
                  Vector(pTS, index), VectorSize(pCB)), vector[i]);
      // Store the best result
      if ((bestNewPot < 0) || (newPot < bestNewPot)) 
        {
        bestNewPot = newPot;
     /* bestNewIndex = index; */
        }
      } 
    currentPot = bestNewPot;
    for (i= 0; i < n; i++)
      vector[i] = min(VectorDist(Vector(pTS, i), 
                  Vector(pTS, index), VectorSize(pCB)), vector[i]);
    CopyVector(Vector(pTS, index), Vector(pCB, centerCount), VectorSize(pCB));
  }
  
  free(vector);
}


/*-------------------------------------------------------------------*/
/* Random selection by Luxburg:                                      */
/* von Luxburg, Clustering stability: an overview                    */
/* Foundations and Trends in Machine Learning, 2010                  */
/*-------------------------------------------------------------------*/


void  LuxburgInitialCentroids(TRAININGSET *pTS, CODEBOOK *pCB)
{
  int           i, new, k, L, size;
  CODEBOOK      CB2;
  PARTITIONING  P, P2;
  llong         distance[BookSize(pTS)];
  double        c=0, err=0;

  k    = BookSize(pCB);
  L    = (int) (k * log(k)/log(2));
  size = BookSize(pTS)/L;

  PrintMessage("N=%i  L=%i  k=%i \n", BookSize(pTS), L, k);

  // Step 1: L = k*log(k) initial centroids.

  CreateNewCodebook(&CB2, L, pTS);
  InitializeSolution(&P2, &CB2, pTS, L);   
  err = GenerateInitialSolution(&P2, &CB2, pTS, NO);
  PrintMessage("Random codebook (%i)\n", L);
  PrintCodebook(&CB2);

//  CreateNewCodebook(&CB2, L, pTS);
//  RandomCodebook(pTS, &CB2);

  // Step 2: One iteration of k-means

  PrintMessage("One iteration of k-means...\n");
  CalculateDistances(pTS, &CB2, &P2, distance);
  KMeans(&P2, &CB2, pTS, distance, 1, NO, 0, c);
  PrintCodebook(&CB2);

  // Step 3: Remove small centroids (size<N/L)

  PrintMessage("Remove small centroids...\n");
  for( i=0; i<BookSize(&CB2); i++)
     {
     if( VectorFreq(&CB2,i) < size )
        {
        PrintMessage("Remove vector (%i):  freq=%i < limit=%i\n", i, VectorFreq(&CB2,i), size);
        RemoveFromCodebook(&CB2, i);
        i--;   // If removed, another centroid will occupy the slot.
        }
     }
  PrintCodebook(&CB2);

  // Step 4: Heuristic selection aiming at even distribution
  // This part is still buggy, CB structures do not work this way... :-/

  PrintMessage("Heuristic selection initialization...\n");
  // First vector randomly
  ChangeCodebookSize(pCB, 1);
  new = IRZ(BookSize(&CB2));
  CopyVector( Vector(&CB2,new), Vector(pCB,0), VectorSize(pTS));
  CreateNewPartitioning(&P, pTS, 1);
  GenerateOptimalPartitioning(pTS, pCB, &P);

  // Next vector the one with furthest from its centroid
  PrintMessage("Ready to create %i size book...\n", k);
  for( i=1; i<k; i++ )
     {
     new = VectorCausingBiggestError(pTS, pCB, &P, i);
     IncreaseCodebookSize(pCB, BookSize(pCB)+1);
     IncreaseNumberOfPartitions(&P, PartitionCount(&P)+1);
     CopyVector( Vector(pTS,new), Vector(pCB,i), VectorSize(pTS));
     // Inefficient suboptimal. Should be repartition.
     GenerateOptimalPartitioning(pTS, pCB, &P);
     PrintMessage("Vector %i selected\n", i);
     }
  PrintCodebook(pCB);

}


/*-------------------------------------------------------------------*/


int SelectRandomDataObject(CODEBOOK *pCB, TRAININGSET *pTS)
{
  int i, j, count = 0;
  int ok;

  do 
    {
    count++;

    /* random number generator must be initialized! */
    j = IRZ(BookSize(pTS));

    /* eliminate duplicates */
    ok = 1;
    for (i = 0; i < BookSize(pCB); i++) 
      {
      if (EqualVectors(Vector(pCB, i), Vector(pTS, j), VectorSize(pTS)))
        {
        ok = 0;
        }
      }
  } 
  while (!ok && (count <= BookSize(pTS)));   /* fixed 25.01.2005 */

  return j;
}


/*-------------------------------------------------------------------*/
/* random number generator must be initialized! */


void RandomSwap(CODEBOOK *pCB, TRAININGSET *pTS, int *j, int deterministic, 
                int quietLevel)
{
  int i;

  if (!deterministic)
    {
    *j = IRZ(BookSize(pCB));
    }

  i = SelectRandomDataObject(pCB, pTS);

  CopyVector(Vector(pTS, i), Vector(pCB, *j), VectorSize(pTS));
  if (quietLevel >= 5)  PrintMessage("Random Swap done: x=%i  c=%i \n", i, *j);
}



/*-------------------------------------------------------------------*/


void LocalRepartition(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS, int j, 
double time, int quietLevel)
{
  if (quietLevel >= 5)  PrintMessage("Local repartition of vector %i \n", j);

  /* object rejection; maps points from a cluster to their nearest cluster */
  LocalRepartitioningGeneral(pTS, pCB, pP, j, EUCLIDEANSQ);

  /* object attraction; moves vectors from their old partitions to
     a the cluster j if its centroid is closer */
  RepartitionDueToNewVectorGeneral(pTS, pCB, pP, j, EUCLIDEANSQ);

  if (quietLevel >= 3)  PrintMessage("RepartitionTime= %f   ", GetClock(time));
} 


/*-------------------------------------------------------------------*/
// AKTIVITEETIN PÄIVITTÄMINEN TULEE TÄNNE

/* generates optimal codebook with respect to a given partitioning */
void OptimalRepresentatives(PARTITIONING *pP, TRAININGSET *pTS, CODEBOOK *pCB, 
int *active, llong *cdist, int *activeCount, int travellerSearch)
{
  int i, j;
  VECTORTYPE v;

  j = 0;
  v = CreateEmptyVector(VectorSize(pCB));

  for(i = 0; i < BookSize(pCB); i++)
    {  
    if (CCFreq(pP, i) > 0)
      {
      CopyVector(Vector(pCB, i), v, VectorSize(pCB));
      /* calculate mean values for centroid */
      PartitionCentroid(pP, i, &Node(pCB, i));
      /* if centroid changed, cluster is active */
      if (CompareVectors(Vector(pCB, i), v, VectorSize(pCB)) != 0)
        {
	if(travellerSearch) 
          {
	  /* calculate the distance centroid moved */
          cdist[i] = VectorDistance(v, Vector(pCB, i), VectorSize(pTS), MAXLLONG, EUCLIDEANSQ);
          }
        active[j] = i;
        j++;
        }
      }
    else
      {
      VectorFreq(pCB, i) = 0;
      }
    }

  FreeVector(v);
  (*activeCount) = j;
}  


/*-------------------------------------------------------------------*/
/* arr must be sorted ascending order! */


int BinarySearch(int *arr, int size, int key)
{
  int top, bottom, middle;

  top = 0;
  bottom = size - 1;
  middle = (top + bottom) / 2;

  do 
    {
    if (arr[middle] < key)     top    = middle + 1;
    else                       bottom = middle;
    middle = (top + bottom) / 2;
    } 
  while (top < bottom);

  if (arr[middle] == key)    return middle;
  else                       return -1;
}


/*-------------------------------------------------------------------*/
/* generates optimal partitioning with respect to a given codebook */
// AKTIIVINEN-PASIIVINEN VEKTORI MUUTOS


void OptimalPartition(CODEBOOK *pCB, TRAININGSET *pTS, PARTITIONING *pP, 
int *active, llong *cdist, int activeCount, llong *distance, int quietLevel)
{
  int i, j, k;
  int nearest;
  llong error, dist;
  CODEBOOK CBact;
  
  if (quietLevel >= 5)  PrintMessage("\n Optimal Partition starts. ActiveCount=%i..\n", activeCount);

  /* all vectors are static; there is nothing to do! */
  if (activeCount < 1) return;

  /* creating subcodebook (active clusters) */
  if (quietLevel >= 5)  PrintMessage("Creating subcodebook...");
  CreateNewCodebook(&CBact, activeCount, pTS);
  for (i = 0; i < activeCount; i++) 
    {
    CopyVector(Vector(pCB, active[i]), Vector(&CBact, i), VectorSize(pCB));
    }
  if (quietLevel >= 5)  PrintMessage("Done.\n");
  
  if (quietLevel >= 5)  PrintMessage("Looping ... ");
  for(i = 0; i < BookSize(pTS); i++)
     {
     if (quietLevel >= 5)  PrintMessage(" %i ", i);
     j     = Map(pP, i);
     k     = BinarySearch(active, activeCount, j);
     dist  = VectorDistance(Vector(pTS, i), Vector(pCB, j), VectorSize(pTS), MAXLLONG, EUCLIDEANSQ); 
     
     // static vector - search subcodebook
     if (k < 0)  
       {
       nearest = FindNearestVector(&Node(pTS,i), &CBact, &error, 0, EUCLIDEANSQ);
       nearest = (error < dist) ? active[nearest] : j;
       }
     // active vector, centroid moved closer - search subcodebook
     else if (dist < distance[i])  
       {
       nearest = FindNearestVector(&Node(pTS,i), &CBact, &error, k, EUCLIDEANSQ);
       nearest = active[nearest];
       } 
     // active vector, centroid moved farther - FULL search
     else  
       {
       nearest = FindNearestVector(&Node(pTS,i), pCB, &error, j, EUCLIDEANSQ);
       }
     
     if (nearest != j)  
       {
       /* closer cluster was found */
       ChangePartition(pTS, pP, nearest, i);
       distance[i] = error;
       } 
     else 
       {
       distance[i] = dist;
       }
    }

  FreeCodebook(&CBact);
  
  if (quietLevel >= 5)  PrintMessage("Optimal Partition ended.\n");
}


/*-------------------------------------------------------------------*/
/* generates optimal partitioning with respect to a given codebook */
// AKTIIVINEN-PASIIVINEN VEKTORI MUUTOS


void OptimalPartitionTraveller(CODEBOOK *pCB, TRAININGSET *pTS, PARTITIONING *pP, 
int *active, llong *cdist, int activeCount, llong *distance, int quietLevel)
{
  int i, j, k, l;
  int nearest;
  llong error, dist;
  CODEBOOK CBact;
  
  if (quietLevel >= 5)  PrintMessage("\n Optimal Partition (Traveller variant) starts. ActiveCount=%i..\n", activeCount);

  /* all vectors are static; there is nothing to do */
  if (activeCount < 1)  return;

  /* traveller codebook variables */
  CODEBOOK CBactTrvs[activeCount];
  int trvCount[activeCount];
  int trvArray[activeCount][BookSize(pCB)];
  int ptrv[BookSize(pCB)];
 
  if (quietLevel >= 5)  PrintMessage("Traveller arrays created.\n");

  /* creating subcodebook (active clusters) */
  if (quietLevel >= 5)  PrintMessage("Creating subcodebook...");
  CreateNewCodebook(&CBact, activeCount, pTS);
  for (i = 0; i < activeCount; i++) {
    CopyVector(Vector(pCB, active[i]), Vector(&CBact, i), VectorSize(pCB));
  if (quietLevel >= 5)  PrintMessage("Done.\n");
  
  /* Creating traveller codebooks */
  for (i = 0; i < activeCount; i++) 
      {
      /* find centroids among active centroieds whose move distance is greater than current centroids */
      trvCount[i] = 0;
      for (l = 0; l < activeCount; l++) 
          {
          if(active[i] == active[l] || cdist[active[i]] < cdist[active[l]]) 
              {
              trvArray[i][trvCount[i]++] = active[l];
              }
          }
      /* create traveller codebook */
      CreateNewCodebook(&CBactTrvs[i], trvCount[i], pTS);
      for (l = 0; l < trvCount[i]; l++)  
          {
          CopyVector(Vector(pCB, trvArray[i][l]), Vector(&CBactTrvs[i], l), VectorSize(pCB));
          }
      ptrv[active[i]] = i;
      }

  if (quietLevel >= 5)  PrintMessage("Looping ... ");
  for(i = 0; i < BookSize(pTS); i++)
     {
     if (quietLevel >= 5)  PrintMessage(" %i ", i);
     j     = Map(pP, i);
     k     = BinarySearch(active, activeCount, j);
     dist  = VectorDistance(Vector(pTS, i), Vector(pCB, j), VectorSize(pTS), MAXLLONG, EUCLIDEANSQ); 
     
     if (k < 0)  /* static vector */
       {
       // search subcodebook
       nearest = FindNearestVector(&Node(pTS,i), &CBact, &error, 0, EUCLIDEANSQ);
       nearest = (error < dist) ? active[nearest] : j;
       }
     else if (dist < distance[i])  /* active vector, centroid moved closer */
       {
       // search traveller codebook
       k = BinarySearch(trvArray[ptrv[j]], trvCount[ptrv[j]], j);
       nearest = FindNearestVector(&Node(pTS,i), &CBactTrvs[ptrv[j]], &error, k, EUCLIDEANSQ);
       nearest = trvArray[ptrv[j]][nearest];
       } 
     
     else  /* active vector, centroid moved farther */
       {
       // search full codebook
       nearest = FindNearestVector(&Node(pTS,i), pCB, &error, j, EUCLIDEANSQ);
       }
     
     if (nearest != j)  
       {
       /* closer cluster was found */
       ChangePartition(pTS, pP, nearest, i);
       distance[i] = error;
       } 
     else 
       {
       distance[i] = dist;
       }
  }

  FreeCodebook(&CBact);
  
  /* free traveller codebooks */
  for(i = 0; i < activeCount; i++) 
     {
     FreeCodebook(&CBactTrvs[i]);
     }
  }

  if (quietLevel >= 5)  PrintMessage("Optimal Partition Traveller variant ended.\n");
}


/*-------------------------------------------------------------------*/
/* fast K-means implementation (uses activity detection method) */


void KMeans(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS, llong *distance, 
int iter, int travellerSearch, int quietLevel, double time) 
{

  double starttime = GetClock(time);
  
  int     i, activeCount;
  int     active[BookSize(pCB)];
  llong   cdist[BookSize(pCB)];

  CalculateDistances(pTS, pCB, pP, distance);

  double inittime = GetClock(time) - starttime;
  
  /* performs iter K-means iterations */
  for (i = 0; i < iter; i++)
    {
    /* OptimalRepresentatives-operation should be before 
       OptimalPartition-operation, because we have previously tuned 
       partition with LocalRepartition-operation */ 
    OptimalRepresentatives(pP, pTS, pCB, active, cdist, &activeCount, travellerSearch);
    if(travellerSearch)
       {
       OptimalPartitionTraveller(pCB, pTS, pP, active, cdist, activeCount, distance, quietLevel);
       }
    else
       {
       OptimalPartition(pCB, pTS, pP, active, cdist, activeCount, distance, quietLevel);
       }

    if (quietLevel >= 3)  
      {
      PrintIterationActivity(GetClock(time), i, activeCount, BookSize(pCB), quietLevel);
      }
    }

  if ((quietLevel >= 4) && iter > 0) 
     {
     PrintIterationKMSummary(GetClock(time)-starttime, inittime);
     }
}


/*-------------------------------------------------------------------*/


llong ObjectiveFunction(PARTITIONING *pP, CODEBOOK *pCB, TRAININGSET *pTS)
{
  llong sum = 0;
  int i, j;

  /* sum of squared distances of the data object to their 
     cluster representatives */
  for (i = 0; i < BookSize(pTS); i++) 
    {
    j = Map(pP, i);
    sum += VectorDistance(Vector(pTS, i), Vector(pCB, j), 
           VectorSize(pTS), MAXLLONG, EUCLIDEANSQ) * VectorFreq(pTS, i); 
    }

  return sum;
}


/* -------------------------------------------------------------------- */
/* Calculates data objects current distances to their cluster centroids */
/* -------------------------------------------------------------------- */


void CalculateDistances(TRAININGSET *pTS, CODEBOOK *pCB, PARTITIONING *pP, llong *distance)
{
  int i, j;

  for (i = 0; i < BookSize(pTS); i++) 
    {
    j = Map(pP, i);
    distance[i] = VectorDistance(Vector(pTS, i), Vector(pCB, j), VectorSize(pTS), MAXLLONG, EUCLIDEANSQ); 
    }
} 


/*-------------------------------------------------------------------*/


int FindSecondNearestVector(BOOKNODE *node, CODEBOOK *pCB, 
                            int firstIndex, llong *secondError)
{
  int   i;
  int   secondIndex;
  llong e;

  secondIndex = -1;
  *secondError = MAXLLONG;

  for(i = 0; i < BookSize(pCB); i++)
    {
    e = VectorDistance(Vector(pCB,i), node->vector, VectorSize(pCB), 
        *secondError, EUCLIDEANSQ);

      if ((e < *secondError) && (i != firstIndex))
    {
      *secondError = e;
      secondIndex  = i;
      }
    }
  return secondIndex;
}


/*-------------------------------------------------------------------*/
/* selects deterministicly, which cluster centroid to swap. one that 
   increases objective function (MSE) least, if removed, is selected. */

int SelectClusterToBeSwapped(TRAININGSET *pTS, CODEBOOK *pCB, 
                             PARTITIONING *pP, llong *distance)
{
  int i, j, k, min;
  llong error;
  llong priError[BookSize(pCB)];  /* current error; data objects are in 
                                     their primary (closest) cluster) */
  llong secError[BookSize(pCB)];  /* error after partition is removed and 
                                     data objects are repartitioned; data 
                                     objects are in their secondary 
                                     (second closest) cluster */

  /* initializing */
  for (i = 0; i < BookSize(pCB); i++) 
    {
    priError[i] = 0;
    secError[i] = 0;
    }

  /* calculating primary and secondary cluster errors */
  for (i = 0; i < BookSize(pTS); i++) 
    {
    j = Map(pP, i);
    k = FindSecondNearestVector(&Node(pTS,i), pCB, j, &error);
    /* k will not be stored, only the return error value is used */

    priError[j] += distance[i] * VectorFreq(pTS, i);
    secError[j] += error * VectorFreq(pTS, i);    
    }

  /* finding cluster that increases objective function least */
  min = -1;
  error = MAXLLONG;
  for (j = 0; j < BookSize(pCB); j++) 
    {
    if ((secError[j] - priError[j]) < error)
      {
      min = j;
      error = secError[j] - priError[j];
      }    
    }

  return min;
}


/*-------------------------------------------------------------------*/


char* RSInfo(void)
{
  char* p;
  int len;
  
  len = strlen(ProgName)+strlen(VersionNumber)+strlen(LastUpdated)+4;  
  p   = (char*) malloc(len*sizeof(char));
  
  if (!p) 
    {
    ErrorMessage("ERROR: Allocating memory failed!\n");
    ExitProcessing(FATAL_ERROR);
    }
 
  sprintf(p, "%s\t%s\t%s", ProgName, VersionNumber, LastUpdated);
 
  return p;
}


/*-------------------------------------------------------------------*/
