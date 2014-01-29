/* ============================================================================
**
** neuron GL (c) Eihis 2013-2014
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License
** as published by the Free Software Foundation; either version 2
** of the License, or (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
**
** ========================================================================= */
//
// primusrun ./neuronGL


#include	<stdlib.h>
#include	<GL/glut.h>
#include	<math.h>

// by me for neuron
#include	<stdio.h>

// ****** start neuron functions
//
// neuron setups
//
//// Data dependent settings ////


#define _max_hidden 1024
#define _max_inputs 1024
#define _max_patterns 128

//// User defineable settings ////
int numHidden=8;
int numInputs=3;		// 2 input neurons, +1 for bias 
int numPatterns=30;

const int numEpochs =1000;
const double LR_IH = 0.5;
const double LR_HO = 0.05;


//// functions ////
void initWeights(void);
void initData(void);
void calcNet(void);
void WeightChangesHO(void);
void WeightChangesIH(void);

void calcOverallError(void);

void displayResults(void);
double getRand(void);
unsigned int frand_a_b(double a, double b);


//// variables ////
int patNum = 0;
double errThisPat = 0.0;
double outPred = 0.0;
double RMSerror = 0.0;
double bias_value = 1.0 ;			// variable bias
//
//
// my training : V vers le haut, 30 echantillons
	double inputs[30][3]={{0.10, 0.03, 0},{0.11, 0.11, 0},{0.11, 0.82, 0},
	{0.13, 0.17, 0},{0.20, 0.81, 0},{0.21, 0.57, 1},
	{0.25, 0.52, 1},{0.26, 0.48, 1},{0.28, 0.17, 1},{0.28, 0.45, 1},{0.37, 0.28, 1},{0.41, 0.92, 0},{0.43, 0.04, 1},{0.44, 0.55, 1},{0.47, 0.84, 0},{0.50, 0.36, 1},
	{0.51, 0.96, 0},{0.56, 0.62, 0},{0.65, 0.01, 0},{0.67, 0.50, 1},{0.73, 0.05, 1},{0.73, 0.90, 0},{0.73, 0.99, 0},{0.78, 0.01, 1},{0.83, 0.62, 0},{0.86, 0.42, 1},
	{0.86, 0.91, 0},{0.89, 0.12, 1},{0.95, 0.15, 1},{0.98, 0.73, 0}};
	
// copied to trainInputs // trainInputs[30]=input zone

// the outputs of the hidden neurons
double hiddenVal[_max_hidden];	// mximum hidden neuron count : 1024

// the weights
double weightsIH[_max_inputs][_max_hidden];
double weightsHO[_max_hidden];

// the data
//
double trainInputs[_max_patterns+1][_max_inputs];
double trainOutput[_max_patterns];
//
//
// ******** end neuron functions

/*
** Function called to update rendering
*/
static float alpha_pump= 5.0;
static float alpha = 0;
//

void DisplayFunc(void)
{
    int f;
    float net_x;
    float net_y;
    float net_x_step;
    float net_y_step;
    //
	static float zoom_z = 0.0;
	static float pump_z = 0;
	

  /* Clear the buffer, clear the matrix */
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  /* A step backward, then spin the cube */
  zoom_z+=pump_z;
  
  glTranslatef(0, 0, -10.0+ zoom_z);
  glRotatef(30, 2, 0, 0);
  //if (zoom_z>2) { pump_z=-pump_z;}
  //if (zoom_z< -2) {pump_z=-pump_z;}
  glRotatef(alpha, 0, 1, 0);
  // ========== generating pattern 3d plane : Training      ====================
   glBegin(GL_LINES);
  for (f=0;f<numPatterns;f++)
  {
	 glColor3f(0, 0, 0); // line start
     glVertex3f( (trainInputs[f][0]*1),0,(trainInputs[f][1]*1));
     outPred=trainOutput[f]; 
     glColor3f((outPred+1)/2.0,0, 1-((outPred+1)/2.0));
     glVertex3f( (trainInputs[f][0]*1),trainOutput[f]/6.0,(trainInputs[f][1]*1));
  }
  glEnd();
  // ========== generating pattern 3d plane : Network answer ====================
  net_x_step=0.05;
  net_y_step=0.05;
  patNum=30;
  trainInputs[30][2]=bias_value;										// fix up live bias_value
  //
  glBegin(GL_QUADS); 
  for (net_y=-1;net_y<1;net_y+=net_y_step)
  {
	      
		for(net_x=-1;net_x<1;net_x+=net_x_step)
		{
			trainInputs[30][0]=net_x;
			trainInputs[30][1]=net_y;			
			calcNet();
			glColor3f((outPred+1)/2.0,0, 1-((outPred+1)/2.0));
			glVertex3f(net_x,(outPred)/6.0,net_y); // vertex1
			//
			trainInputs[30][0]=net_x+net_x_step;
			trainInputs[30][1]=net_y;			
			calcNet();
			glColor3f((outPred+1)/2.0,0, 1-((outPred+1)/2.0));
			glVertex3f(net_x+net_x_step,(outPred)/6.0,net_y);
			//
			trainInputs[30][0]=net_x+net_x_step;
			trainInputs[30][1]=net_y+net_y_step;			
			calcNet();
			glColor3f((outPred+1)/2.0,0, 1-((outPred+1)/2.0));
			glVertex3f(net_x+net_x_step,(outPred)/6.0,net_y+net_y_step);
			//
			trainInputs[30][0]=net_x;
			trainInputs[30][1]=net_y+net_y_step;			
			calcNet();
			glColor3f((outPred+1)/2.0,0, 1-((outPred+1)/2.0));
			glVertex3f(net_x,(outPred)/6.0,net_y+net_y_step);
		}
		
  }
  glEnd();
  /* End */
  glFlush();
  glutSwapBuffers();
  //
  alpha+=0.2;
  /* Update again and again */
  glutPostRedisplay();
}

/*
** Function called when the window is created or resized
*/
void		ReshapeFunc(int width, int height)
{
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();
  gluPerspective(20, width / (float) height, 5, 15);
  glViewport(0, 0, width, height);

  glMatrixMode(GL_MODELVIEW);
  glutPostRedisplay();
}

/*
** Function called when a key is hit
*/
void KeyboardFunc(unsigned char key, int x, int y)
{
  int foo;

  foo = x + y; /* Has no effect: just to avoid a warning */
  //if ('q' == key || 'Q' == key || 27 == key) {  exit(0);}
  switch (key)
  {
	// spin rotation invert
	case 13:
		alpha_pump=-alpha_pump;
	break;
	case '4':
		/* Rotate a bit more */
		alpha = alpha - alpha_pump;
	break;
	case '6':
		/* Rotate a bit more */
		alpha = alpha + alpha_pump;
	break;
	case '8':
		bias_value+=0.05;
	break;
	case '2':
		bias_value-=0.05;
	break;
	// exit
	case 'q':
	case 'Q':
	case 27:
		exit(0);
	break;
	// headaches
	default:
	break;
  }
}


int		main(int argc, char **argv)
{
	int j,i;		
  /* Creation of the window */
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );//A=;// | GLUT_DEPTH);//glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  
  glutInitWindowSize(640, 480);
  glutCreateWindow("neuronGL :: sideme / EihiS");
  glClearColor(0.1f,0.1f,0.1f,0.9f);
  //
  glEnable(GL_DEPTH_TEST);
  //
  // now inits/ setup neuron
  printf("Init Data & Weight (randomize)\n");
  initData();
  initWeights();
  //
  printf("Training..\n");
  for(j = 0;j <= numEpochs;j++)
	{
		for(i = 0;i<numPatterns;i++)
			{
				// bkg show
				//calculate the current network output
				//select a pattern at random
				patNum = rand()%numPatterns;
				calcNet();
				//change network weights
				WeightChangesHO();
				WeightChangesIH();
				//
			}

		calcOverallError();
		printf("e =%d errRMS=%f \n",j,RMSerror);
		
		}
  displayResults();
  /* Declaration of the callbacks */
  glutDisplayFunc(&DisplayFunc);
  glutReshapeFunc(&ReshapeFunc);
  glutKeyboardFunc(&KeyboardFunc);

  /* Loop */
  glutMainLoop();

  /* Never reached */
  return 0;
}

//

unsigned int frand_a_b(double a, double b)
{
    return (unsigned int) (( rand()/(double)RAND_MAX ) * (b-a) + a);
}






//
// neuron fonctions
//
//
// calculates the network output
void calcNet(void)
{
    //calculate the outputs of the hidden neurons
    //the hidden neurons are tanh
	int j;
    int i = 0;
    for(i = 0;i<numHidden;i++)
    {
	  hiddenVal[i] = 0.0;

        for(j = 0;j<numInputs;j++)
        {
			hiddenVal[i] = hiddenVal[i] + (trainInputs[patNum][j] * weightsIH[j][i]);
        }
        hiddenVal[i] = tanh(hiddenVal[i]);
    }

   //calculate the output of the network
   //the output neuron is linear
   outPred = 0.0;

   for(i = 0;i<numHidden;i++)
   {
    outPred = outPred + hiddenVal[i] * weightsHO[i];
   }
    //calculate the error
    errThisPat = outPred - trainOutput[patNum];

}


//
//adjust the weights hidden-output
void WeightChangesHO(void)
{
	int k;
   for(k = 0;k<numHidden;k++)
   {
    double weightChange = LR_HO * errThisPat * hiddenVal[k];
    weightsHO[k] = weightsHO[k] - weightChange;

    //regularisation on the output weights
    if (weightsHO[k] < -5)
    {
     weightsHO[k] = -5;
    }
    else if (weightsHO[k] > 5)
    {
     weightsHO[k] = 5;
    }
   }

 }


//
// adjust the weights input-hidden
void WeightChangesIH(void)
{
  int i,k;
  double weightChange;
  for(i = 0;i<numHidden;i++)
  {
   for(k = 0;k<numInputs;k++)
   {
    double x = 1 - (hiddenVal[i] * hiddenVal[i]);
    x = x * weightsHO[i] * errThisPat * LR_IH;
    x = x * trainInputs[patNum][k];
    weightChange = x;
    weightsIH[k][i] = weightsIH[k][i] - weightChange;
   }
  }

}


//
// generates a random number
double getRand(void)
{
 return ((double)rand())/(double)RAND_MAX;
}



//
// set weights to random numbers 
void initWeights(void)
{
 int i,j;
 for(j = 0;j<numHidden;j++)
 {
    weightsHO[j] = (getRand() - 0.5)/2;
    for(i = 0;i<numInputs;i++)
    {
     weightsIH[i][j] = (getRand() - 0.5)/5;
     //printf("Weight = %f\n", weightsIH[i][j]);
    }
  }

}


//
// read in the data
void initData(void)
{
    
	//double trainInputs[30][3];
	//double trainOutput[30];
	int i;
	//
	printf("initialising data\n");
	for (i=0;i<30;i++) // crée la liste d'entrainement
	{
		trainOutput[i]=(inputs[i][2]-0.5)*2;	// output learn : -1..1
		//
		trainInputs[i][0]=(inputs[i][0]-0.5)*2;
		trainInputs[i][1]=(inputs[i][1]-0.5)*2;
		trainInputs[i][2]=bias_value;			// fixed bias
	}
	trainInputs[30][0]=0;		// pour test du reseau de neurones en live
	trainInputs[30][1]=0;
	trainInputs[30][2]=bias_value;	

}


//
// display results
void displayResults(void)
{
	int i;
	//
	// top displays
	//
	printf("Learn patterns:%d \n",numPatterns);
	printf("Input neurons:%d \n",numInputs);
	printf("Hidden neurons:%d \n",numHidden);
	//
	//printf("pat[]\0");
	//text(50,0,"act\0");
	//text(100,0,"model\0");
	 for(i = 0;i<numPatterns;i++)
	 {
		  patNum = i;
		  calcNet();
		  
		  //text(
		  //number(0,16+i*16,patNum+1);
		  //number_f(50,16+i*16,trainOutput[patNum]);
		  //number_f(100,16+i*16,outPred);
		  //
		  printf("pat:%d  act:%f   model:%f \n",patNum+1,trainOutput[patNum],outPred);
	 }
}

//
// calculate the overall error
void calcOverallError(void)
{
	int i;
     RMSerror = 0.0;
     for(i = 0;i<numPatterns;i++)
        {
         patNum = i;
         //calcNet();
         RMSerror = RMSerror + (errThisPat * errThisPat);
        }
     RMSerror = RMSerror/numPatterns;
     RMSerror = sqrt(RMSerror);
}


