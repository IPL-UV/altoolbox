const char *help = "\
Multi-class SVM with interpolation option\n\
Author: Fred Ratle \n\
Based on SVMTorch Multi III (c) Trebolloc & Co 2002\n\
Last modified: 19.07.2007 \n\
This program will train several SVMs, for classification with\n\
more than two classes, using a one-against-all approach.\n\
It uses a gaussian kernel (default) or a polynomial kernel. \n\
Four modes are available: train, kfold, test\n\
and interpolation modes \n";

#include "MatDataSet.h"
#include "OneHotClassFormat.h"
#include "ClassMeasurer.h"
#include "MSEMeasurer.h"
#include "QCTrainer.h"
#include "CmdLine.h"
#include "Random.h"
#include "SVMClassification.h"
#include "DiskXFile.h"
#include "ClassFormatDataSet.h"

using namespace Torch;


//-----------------------------------------------------------------------------
//This function builds a n-dimensional grid with the user-specified limits
//-----------------------------------------------------------------------------
bool build_mat(int nvar, int echant,real* limites[2], real* &ret_mat)
{
    if(limites == NULL || nvar < 0 || echant < 0){
        return false;
    }

    int nb_pts = int(pow(double(echant),nvar));
    int taille = nb_pts*nvar;
    ret_mat = new real[taille];

    int* val = new int[nvar];
    memset(val, 0,nvar*sizeof(int));

    int nb_div = echant-1;
    if(nb_div == 0)
	{
        nb_div = 1;
    }

    for(int i=0; i<nb_pts; ++i)
	{
        for(int j=0; j<nvar; ++j)
		{
            ret_mat[i*nvar+j] = limites[0][j] + real(val[j])/real(nb_div)*(limites[1][j] - limites[0][j]);
            if((i+1) % int(pow(double(echant),j)) ==0)
			{
                ++val[j];
                val[j] %= echant;
            }
        }
    }

    if (val != NULL)
	{
        delete val;
    }

    return true;
}



//---------------------------------------------------
// Function that creates creates the grid datafile
//---------------------------------------------------

void createGrid(int nvar, int echant, char* file, char* file2){
	  real * bounds[2];

	  FILE * fic;
	  fic=fopen(file,"r");
	  if (fic==NULL){
		  error("Cannot open boundary file");
	  }

	  int nb_pts = int(pow(double(echant), nvar));
	  real * mat = new real[nb_pts];

	  for (unsigned i=0; i<2; ++i){
		  bounds[i] = new real[nvar];
		  for (int j=0;j<nvar;++j){
			  fscanf(fic, "%f", &bounds[i][j] );
		  }
	  }
	  fclose(fic);


	  build_mat(nvar,echant,bounds,mat);

	  FILE * fic2;
	  fic2=fopen(file2, "w");
	  if (fic2==NULL){
		  error("Cannot create grid file");
	  }

	  fprintf(fic2, "%d\t", nb_pts );
	  fprintf(fic2, "%d\n", nvar );
	  for (int i=0;i<nb_pts;++i){
		  for (int j=0;j<nvar;++j){
			  fprintf(fic2, "%f\t", mat[i*nvar+j] );
		  }
		  fprintf(fic2, "\n");
	  }
	  fclose(fic2);
}




//-------------------
// main function
//-------------------

void the_main(int argc, char **argv)
{
  char* file;
  char* lfile;
  real c_cst, stdv; // stdv2;
  real accuracy, cache_size;
  int iter_shrink;
  int the_seed;
  int max_load;
  char *dir_name;
  char *model_file;
  bool binary_mode;
  int degree;
  int nvar;
  int echant;
  real a_cst, b_cst;
  //real alpha;
  int n_classes;
  //char* limits;

  Allocator *allocator = new Allocator;

  //=================== The command-line ==========================

  // Construct the command line
  CmdLine cmd;

  // Put the help line at the beginning
  cmd.info(help);

  // Train mode
  cmd.addText("\nArguments:");
  cmd.addSCmdArg("file", &file, "the train file");
  cmd.addSCmdArg("model", &model_file, "the model file");
  cmd.addICmdArg("# classes", &n_classes, "the number of classes", true);

  cmd.addText("\nModel Options:");
  cmd.addRCmdOption("-c", &c_cst, 100., "trade off cst between error/margin");
  cmd.addRCmdOption("-std", &stdv, 10., "the std parameter in the gaussian kernel [exp(-|x-y|^2/std^2)]", true);
//  cmd.addRCmdOption("-std2", &stdv2, -1, "if positive, use a multigaussian kernel with specified std2", true);
//  cmd.addRCmdOption("-alpha", &alpha, 0.5, "the weight in the multigaussian kernel", true);

  cmd.addICmdOption("-degree", &degree, -1, "if positive, use a polynomial kernel [(a xy + b)^d] with the specified degree", true);
  cmd.addRCmdOption("-a", &a_cst, 1., "constant a in the polynomial kernel", true);
  cmd.addRCmdOption("-b", &b_cst, 1., "constant b in the polynomial kernel", true);

  cmd.addText("\nLearning Options:");
  cmd.addRCmdOption("-e", &accuracy, 0.01, "end accuracy");
  cmd.addRCmdOption("-m", &cache_size, 50., "cache size in Mo");
  cmd.addICmdOption("-h", &iter_shrink, 100, "minimal number of iterations before shrinking");

  cmd.addText("\nMisc Options:");
  cmd.addICmdOption("-seed", &the_seed, -1, "the random seed");
  cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for train");
  cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
  cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");


  // validation mode, i.e., labeled data

  cmd.addMasterSwitch("--val");
  cmd.addText("\nArguments:");
  cmd.addSCmdArg("model", &model_file, "the model file");
  cmd.addSCmdArg("file", &file, "the test file");

  cmd.addText("\nMisc Options:");
  cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for test");
  cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
  cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");


  // Test mode, i.e., unlabeled data

  cmd.addMasterSwitch("--test");
  cmd.addText("\nArguments:");
  cmd.addSCmdArg("model", &model_file, "the model file");
  cmd.addSCmdArg("file", &file, "the test file");
 // cmd.addICmdArg("# classes", &n_classes, "the number of classes", true);

  cmd.addText("\nMisc Options:");
  cmd.addICmdOption("-load", &max_load, -1, "max number of examples to load for test");
  cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
  cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");

  // Interpolation mode

  //option for interpolating on a grid
  // the file must have the following structure
  // xmin
  // ymin
  // ...
  // xmax
  // ymax

  cmd.addMasterSwitch("--inter");
  cmd.addText("\nArguments:");
  cmd.addSCmdArg("model", &model_file, "the model file");
  cmd.addSCmdArg("lfile", &lfile, "the interpolation boundaries file");

  cmd.addText("\nMisc Options:");
  cmd.addICmdOption("-nvar", &nvar, 2, "number of variables");
  cmd.addICmdOption("-echant", &echant, 10, "number of samples per variable");
  cmd.addSCmdOption("-dir", &dir_name, ".", "directory to save measures");
  cmd.addBCmdOption("-bin", &binary_mode, false, "binary mode for files");



  // Read the command line
  int mode = cmd.read(argc, argv);

  DiskXFile *model = NULL;
  if(mode > 0)
  {
    model = new(allocator) DiskXFile(model_file, "r");
    cmd.loadXFile(model);
  }

  // If the user didn't give any random seed,
  // generate a random random seed...
  if(mode == 0)
  {
    if(the_seed == -1)
      Random::seed();
    else
      Random::manualSeed((long)the_seed);
  }
  cmd.setWorkingDirectory(dir_name);

  //=================== Create the SVMs... =========================
  Kernel *kernel = NULL;
  if(degree > 0)
    kernel = new(allocator) PolynomialKernel(degree, a_cst, b_cst);
//  else if (stdv2 > 0)
//    kernel = new(allocator) MultiGaussianKernel(1./(stdv*stdv), 1./(stdv2*stdv2), alpha);
  else
    kernel = new(allocator) GaussianKernel(1./(stdv*stdv));

  SVM **svms = (SVM **)allocator->alloc(sizeof(SVM *)*n_classes);
  for(int i = 0; i < n_classes; i++)
  {
    svms[i] = new(allocator) SVMClassification(kernel);

    if(mode == 0)
    {
      svms[i]->setROption("C", c_cst);
      svms[i]->setROption("cache size", cache_size);
    }
  }

  //=================== DataSets & Measurers... ===================

  // Create the training dataset

  char *the_file = new char[100];
  int ntarg;

  if (mode==3){
	//the_file="grid_interpol";
	strcpy(the_file, "grid_interpol");
	createGrid(nvar, echant, lfile, the_file);
	ntarg=0;
	}
  else if(mode==2){
	the_file=file;
	ntarg=0;
  }
  else{
	  the_file=file;
	  ntarg=1;
  }

  MatDataSet *mat_data = new(allocator) MatDataSet(the_file, -1, ntarg, false, max_load, binary_mode);

  // Reload the model in test mode
  if(mode > 0)
  {
    for(int i = 0; i < n_classes; i++)
      svms[i]->loadXFile(model);
  }

  //=================== Let's go... ===============================

  char outname[200];

  // Train
  if(mode == 0)
  {
    DiskXFile model_(model_file, "w");
    cmd.saveXFile(&model_);

    for(int i = 0; i < n_classes; i++)
	{
      message("Training class %d against the others", i);
      QCTrainer trainer(svms[i]);
      trainer.setROption("end accuracy", accuracy);
      trainer.setIOption("iter shrink", iter_shrink);

      Sequence class_labels(n_classes, 1);
      for(int j = 0; j < n_classes; j++)
      {
        if(j == i)
          class_labels.frames[j][0] =  1;
        else
          class_labels.frames[j][0] = -1;
      }
      ClassFormatDataSet data(mat_data, &class_labels);

      trainer.train(&data, NULL);
      message("%d SV with %d at bounds", svms[i]->n_support_vectors, svms[i]->n_support_vectors_bound);
      svms[i]->saveXFile(&model_);
    }


  /*
  // Let's save the constants b, the alphas and the support vectors (in training mode)

  //FILE * out1;
  FILE * out2;
  //FILE * out3;

	for (int i=0;i<n_classes;++i){
	//	snprintf(outname, sizeof(outname), "%s/b%d.dat", dir_name, i);
	//	out1=fopen(outname,"w");
		snprintf(outname, sizeof(outname), "%s/SV%d.dat", dir_name, i);
		out2=fopen(outname,"w");
	//	snprintf(outname, sizeof(outname), "%s/alphas%d.dat", dir_name, i);
	//	out3=fopen(outname,"w");
	// 	fprintf(out1,"%f",svms[i]->b);
	//	fprintf(out2, "The support vectors indices of svm %d are: \n\n", i);
	//	fprintf(out3, "The alphas of svm %d are: \n\n", i);
		for (int j=0;j<(svms[i]->n_support_vectors);++j){
			fprintf(out2, "%d\n",svms[i]->support_vectors[j]);
		}
	//	for (int k=0;k<(svms[i]->n_alpha);++k){
	//		fprintf(out3, "%1.14lf\n",svms[i]->alpha[k]);
	//	}
	//	fprintf(out2,"\n\n");
	//	fprintf(out3,"\n\n");
	//  fclose(out1);
	    fclose(out2);
	//  fclose(out3);
	}
	*/

  }

  // Test, validation or interpolation
  if(mode > 0)
  {
    snprintf(outname, sizeof(outname), "%s/predictions.dat", dir_name);
    FILE * out;
	out=fopen(outname,"w");
	if (out==NULL){
		error("Cannot create output file - exit\n");
	}
    OneHotClassFormat class_format(n_classes);
    int n_errors = 0;

	real *buffer = (real *)allocator->alloc(sizeof(real)*n_classes);

	if (mode==1){
		for(int t = 0; t < mat_data->n_examples; ++t)
		{
			mat_data->setExample(t);
			for(int i = 0; i < n_classes; ++i)
			{
				svms[i]->forward(mat_data->inputs);
				buffer[i] = svms[i]->outputs->frames[0][0];
			}
			int the_class = (int)mat_data->targets->frames[0][0];

			// here we output the predictions of the SVM
			real prediction = class_format.getClass(buffer);
			fprintf(out, "%f", prediction);
			// function output for each SVM
			for (int i=0; i<n_classes; i++) fprintf(out, " %f", buffer[i]);
			fprintf(out, "\n");
		        // and compute the classification error unless in val or inter mode
			if(the_class != class_format.getClass(buffer))
				++n_errors;
		}

		message("%f%% of missclassification. (%d errors)", ((real)n_errors)/((real)mat_data->n_examples)*100., n_errors);
		fclose(out);

		/*
	    snprintf(outname, sizeof(outname), "%s/the_class_err.dat", dir_name);
		out=fopen(outname,"w");
		if (out==NULL){
			error("Cannot create output file - exit\n");
		}
		fprintf(out,"%f\n",((real)n_errors)/((real)mat_data->n_examples)*100.);
		fclose(out);
		*/
	}
	else{
		for(int t = 0; t < mat_data->n_examples; ++t)
		{
			mat_data->setExample(t);
			for(int i = 0; i < n_classes; ++i)
			{
				svms[i]->forward(mat_data->inputs);
				buffer[i] = svms[i]->outputs->frames[0][0];
			}

			// here we output the predictions of the SVM
			real prediction = class_format.getClass(buffer);
			fprintf(out,"%f\n", prediction);
			// function output for each SVM
			for (int i=0; i<n_classes; i++) fprintf(out, " %f", buffer[i]);
			fprintf(out, "\n");
		}

	}

  }

  delete allocator;

  }


  //----------------------------------
  // the main that returns exceptions
  //----------------------------------

int main(int argc, char** argv)
{
  try
  {
    the_main(argc, argv);
  }
  catch (...)
  {
    printf("Exception thrown \n");
  }


  return 0;
}
